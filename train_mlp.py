import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from tqdm import tqdm
import warnings
import random
import os
warnings.filterwarnings('ignore')

# ============================================
# ===== 여기서 학습 설정을 변경하세요 =====
# ============================================
TRAINING_CONFIG = {
    # 기본 설정
    'L': 100,              # 거리 L (km)
    'epochs': 120,         # 훈련 에포크 수
    'batch_size': 64,      # 배치 크기
    # 최적화 설정
    'learning_rate': 0.001,
    'momentum': 0.9,          # SGD momentum
    'weight_decay': 1e-5,
    'dropout_rate': 0.1,
    'loss_scaling': 100
}
# ============================================

def set_seed(seed=42):
    """모든 랜덤성 소스에 대해 시드를 고정하여 재현 가능한 결과 보장"""
    print(f"시드 고정: {seed}")
    
    # Python 내장 random 모듈
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # 환경 변수로도 설정 (일부 라이브러리용)
    os.environ['PYTHONHASHSEED'] = str(seed)

def transform_input_features(X):
    """
    모델 입력 특성에 대해 학습 및 추론 단계에서 동일하게 적용할 변환.
    - Y_0, eps_sec, eps_cor, N: log10 변환 (분포 형상 정규화)
    - 나머지 변수(eta_d, e_d, alpha, zeta): 원본 유지
    - 최종적으로 StandardScaler가 전체를 정규화하므로 ×100/×10 스케일링은 불필요
    """
    X_transformed = np.array(X, dtype=np.float64, copy=True)

    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)

    # 1. eta_d: 원본 유지 (StandardScaler가 정규화)
    # X_transformed[:, 0] = X_transformed[:, 0]  # eta_d

    # 2. Y_0 (1e-7~1e-5): 로그 변환 필요 (10배 단위 변동)
    X_transformed[:, 1] = np.log10(np.clip(X_transformed[:, 1], a_min=1e-20, a_max=None))  # Y_0

    # 3. e_d: 원본 유지 (StandardScaler가 정규화)
    # X_transformed[:, 2] = X_transformed[:, 2]  # e_d

    # 4. alpha: 원본 유지 (StandardScaler가 정규화)
    # X_transformed[:, 3] = X_transformed[:, 3]  # alpha

    # 5. zeta: 원본 유지 (StandardScaler가 정규화)
    # X_transformed[:, 4] = X_transformed[:, 4]  # zeta

    # 6. eps_sec (1e-12~1e-8): 로그 변환 필요 (10배 단위 변동)
    X_transformed[:, 5] = np.log10(np.clip(X_transformed[:, 5], a_min=1e-30, a_max=None))  # eps_sec

    # 7. eps_cor (1e-18~1e-13): 로그 변환 필요 (10배 단위 변동)
    X_transformed[:, 6] = np.log10(np.clip(X_transformed[:, 6], a_min=1e-30, a_max=None))  # eps_cor

    # 8. N (1e9~1e11): 로그 변환 필요 (10배 단위 변동)
    X_transformed[:, 7] = np.log10(np.clip(X_transformed[:, 7], a_min=1.0, a_max=None))  # N

    return X_transformed

def transform_target_outputs(y):
    """
    모델 출력(SKR 포함)에 대해 학습 및 추론 단계에서 동일하게 적용할 변환.
    - p_mu, p_nu, p_vac: 비율(ratio)로 변환 (합=1)
    - SKR: log10 변환 (0 혹은 음수 방지를 위해 최소값 고정)
    """
    y_transformed = np.array(y, dtype=np.float64, copy=True)

    if y_transformed.ndim == 1:
        y_transformed = y_transformed.reshape(1, -1)

    # p_mu, p_nu, p_vac를 비율로 변환 (인덱스 3, 4, 5)
    p_mu = y_transformed[:, 3]
    p_nu = y_transformed[:, 4]
    p_vac = y_transformed[:, 5]
    sum_p = p_mu + p_nu + p_vac
    
    y_transformed[:, 3] = p_mu / sum_p   # r_mu
    y_transformed[:, 4] = p_nu / sum_p    # r_nu
    y_transformed[:, 5] = p_vac / sum_p   # r_vac

    # 9번째 컬럼(SKR)에 로그 변환 적용
    y_transformed[:, -1] = np.log10(np.clip(y_transformed[:, -1], a_min=1e-30, a_max=None))

    return y_transformed

class QKDDataset(Dataset):
    """QKD 데이터셋을 위한 PyTorch Dataset 클래스"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class QKDMLP(nn.Module):
    """QKD 파라미터 최적화를 위한 MLP 신경망 (논문 구조)"""
    def __init__(self, input_size=10, hidden_sizes=[512, 256], output_size=9, dropout_rate=None):
        super(QKDMLP, self).__init__()
        
        # dropout_rate가 None이면 설정값 사용
        if dropout_rate is None:
            dropout_rate = TRAINING_CONFIG['dropout_rate']
        
        # 입력층
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        
        # 은닉층들 (논문: 2개 은닉층)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # 출력층 (9개: 8개 파라미터 + SKR)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # 활성화 함수 (논문: ReLU)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 출력 활성화 함수 (0-1 범위로 제한)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 입력층
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # 은닉층들
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
            x = self.dropout(x)
        
        # 출력층
        x = self.output_layer(x)
        
        # 출력을 0-1 범위로 제한 (논문에서 언급한 대로)
        x = self.sigmoid(x)
        
        return x

class QKDMLPTrainer:
    """QKD MLP 훈련 및 평가를 위한 클래스"""
    
    def __init__(self, config=None):
        """훈련기 초기화"""
        if config is None:
            config = TRAINING_CONFIG
        self.config = config
        
        self.device = torch.device('cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        # 모델 초기화 (dropout_rate는 설정값 사용)
        # L과 e_0는 고정값이므로 입력에서 제외 (8개 입력)
        self.model = QKDMLP(input_size=8, hidden_sizes=[512, 256], output_size=9, 
                           dropout_rate=config['dropout_rate'])
        self.model.to(self.device)
        
        # 옵티마이저 및 손실 함수 (설정값 사용)
        self.optimizer = optim.SGD(self.model.parameters(), 
                                  lr=config['learning_rate'], 
                                  momentum=config['momentum'],
                                  weight_decay=config['weight_decay'])
        self.criterion = nn.MSELoss()  # 논문에서 사용한 MSE Loss
        
        # 스케일러들
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        
        # 파라미터별 가중치 설정 (SKR에 높은 가중치)
        # ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        self.param_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0]).to(self.device)
        print(f"파라미터 가중치: SKR={self.param_weights[-1]:.1f}x, 나머지={self.param_weights[0]:.1f}x")
        
    def load_data(self, csv_path):
        """CSV 파일에서 데이터 로드"""
        print(f"데이터 로드 중: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 입력 파라미터 (8개: L과 e_0는 고정값이므로 제외)
        # L은 각 데이터셋에서 고정값이므로 모델 학습 시 입력에서 제외
        input_columns = ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
        
        # 출력 파라미터 (9개: 8개 파라미터 + SKR)
        output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        
        # 데이터 추출
        X = df[input_columns].values
        y = df[output_columns].values
        
        print(f"데이터 크기: {X.shape[0]} 샘플")
        print(f"입력 차원: {X.shape[1]}, 출력 차원: {y.shape[1]}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """데이터 전처리 - 모든 변수에 적절한 변환 적용"""
        print("데이터 전처리 중...")
        
        # 입력 변수별 적절한 변환 적용
        # ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
        # L과 e_0는 고정값이므로 입력에서 제외
        X_transformed = transform_input_features(X)
        
        print("입력 변환 적용:")
        print("  - Y_0, eps_sec, eps_cor, N: log10(x) 변환")
        print("  - eta_d, e_d, alpha, zeta: 원본 유지 (StandardScaler로 정규화)")
        
        # 입력 데이터 정규화 (StandardScaler)
        X_scaled = self.feature_scaler.fit_transform(X_transformed)
        
        # 출력 데이터 전처리 - p_mu/p_nu/p_vac를 비율로 변환, SKR에 로그 변환 적용
        print("출력 변환 적용:")
        print("  - p_mu, p_nu, p_vac: 비율(ratio)로 변환 (합=1)")
        print("  - SKR: log10(x) 변환")
        y_transformed = transform_target_outputs(y)
        y_scaled = self.target_scaler.fit_transform(y_transformed)
        
        return X_scaled, y_scaled
    
    def weighted_loss(self, output, target):
        """파라미터별 가중치가 적용된 MSE 손실 함수"""
        # 각 파라미터별 MSE 계산
        param_losses = torch.mean((output - target) ** 2, dim=0)  # shape: (9,)
        
        # 가중치 적용
        weighted_losses = param_losses * self.param_weights
        
        # 전체 손실은 가중 평균 (설정값 사용)
        total_loss = torch.sum(weighted_losses) * self.config['loss_scaling']
        
        return total_loss
    
    def create_data_loaders(self, X_train, y_train, batch_size=None):
        """DataLoader 생성 (훈련용만)"""
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        train_dataset = QKDDataset(X_train, y_train)
        
        # 시드 고정을 위해 generator 사용
        generator = torch.Generator()
        generator.manual_seed(42)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            generator=generator,  # 재현 가능한 셔플링
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)  # 멀티프로세싱 시드
        )
        
        return train_loader
    
    def train_epoch(self, train_loader):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 가중치가 적용된 손실 함수 계산
            loss = self.weighted_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.weighted_loss(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, epochs=None):
        """모델 훈련 (검증 없이)"""
        if epochs is None:
            epochs = self.config['epochs']
        print(f"훈련 시작 - 에포크: {epochs}")
        
        best_train_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="훈련 진행"):
            # 훈련
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 최고 모델 저장
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(self.model.state_dict(), 'best_qkd_mlp_model.pth')
            
            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(f"에포크 {epoch+1}/{epochs} - 훈련 손실: {train_loss:.6f}")
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_qkd_mlp_model.pth'))
        print(f"최고 훈련 손실: {best_train_loss:.6f}")
    
    def evaluate(self, test_loader):
        """모델 평가"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        # 역정규화
        predictions_original = self.target_scaler.inverse_transform(predictions)
        targets_original = self.target_scaler.inverse_transform(targets)
        
        # SKR에 대한 역변환 (log10 -> 원본 스케일)
        predictions_original[:, -1] = 10 ** predictions_original[:, -1]  # SKR
        targets_original[:, -1] = 10 ** targets_original[:, -1]  # SKR
        
        # 비율(r_mu, r_nu, r_vac) 재정규화 (합=1 보장)
        # predictions
        r_mu_pred = predictions_original[:, 3]
        r_nu_pred = predictions_original[:, 4]
        r_vac_pred = predictions_original[:, 5]
        sum_r_pred = r_mu_pred + r_nu_pred + r_vac_pred
        predictions_original[:, 3] = r_mu_pred / sum_r_pred
        predictions_original[:, 4] = r_nu_pred / sum_r_pred
        predictions_original[:, 5] = r_vac_pred / sum_r_pred
        
        # targets (역변환된 값도 재정규화)
        r_mu_target = targets_original[:, 3]
        r_nu_target = targets_original[:, 4]
        r_vac_target = targets_original[:, 5]
        sum_r_target = r_mu_target + r_nu_target + r_vac_target
        targets_original[:, 3] = r_mu_target / sum_r_target
        targets_original[:, 4] = r_nu_target / sum_r_target
        targets_original[:, 5] = r_vac_target / sum_r_target
        
        # 평가용 MSE 계산 (원본 스케일)
        eval_mse = np.mean((predictions_original - targets_original) ** 2)
        
        # 각 파라미터별 정확도 계산
        param_names = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        param_errors = {}
        
        for i, param_name in enumerate(param_names):
            param_mse = np.mean((predictions_original[:, i] - targets_original[:, i]) ** 2)
            param_mae = np.mean(np.abs(predictions_original[:, i] - targets_original[:, i]))
            param_errors[param_name] = {'mse': param_mse, 'mae': param_mae}
        
        return {
            'overall_mse': eval_mse,
            'param_errors': param_errors,
            'predictions': predictions_original,
            'targets': targets_original
        }
    
    def predict(self, X):
        """새로운 데이터에 대한 예측"""
        self.model.eval()
        
        # 전처리 (preprocess_data와 동일한 변환 적용)
        X_transformed = transform_input_features(X)
        X_scaled = self.feature_scaler.transform(X_transformed)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # 역정규화
        predictions_original = self.target_scaler.inverse_transform(predictions)
        
        # SKR에 대한 역변환 (log10 -> 원본 스케일)
        predictions_original[:, -1] = 10 ** predictions_original[:, -1]  # SKR
        
        # 비율(r_mu, r_nu, r_vac) 재정규화 (합=1 보장)
        r_mu = predictions_original[:, 3]
        r_nu = predictions_original[:, 4]
        r_vac = predictions_original[:, 5]
        sum_r = r_mu + r_nu + r_vac
        predictions_original[:, 3] = r_mu / sum_r
        predictions_original[:, 4] = r_nu / sum_r
        predictions_original[:, 5] = r_vac / sum_r
        
        return predictions_original
    
    
    def save_model(self, path='qkd_mlp_model.pth'):
        """모델 저장 (PTH 형식)"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_size': 8,  # L과 e_0는 고정값이므로 입력에서 제외
                'hidden_sizes': [512, 256],
                'output_size': 9,
                'dropout_rate': 0.1
            }
        }, path)
        print(f"모델이 {path}에 저장되었습니다.")
    
    def load_model(self, path='qkd_mlp_model.pth'):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"모델이 {path}에서 로드되었습니다.")

def main():
    """메인 실행 함수"""
    # 설정값 사용
    L = TRAINING_CONFIG['L']
    seed = 42  # 고정 시드
    
    # 경로 설정 (L 값 기반)
    train_csv = f"dataset/train_L{L}.csv"
    test_csv = f"dataset/test_L{L}.csv"
    epochs = TRAINING_CONFIG['epochs']
    batch_size = TRAINING_CONFIG['batch_size']
    output_path = f"qkd_mlp_L{L}_E{epochs}_B{batch_size}.pth"
    
    print("=" * 80)
    print(f"QKD MLP 신경망 훈련 (L={L} km)")
    print("Neural Networks for Parameter Optimization in Quantum Key Distribution")
    print("=" * 80)
    
    # 설정 사용
    config = TRAINING_CONFIG.copy()
    
    # 재현 가능한 결과를 위한 시드 고정
    set_seed(seed)
    
    # 훈련기 초기화
    trainer = QKDMLPTrainer(config=config)
    
    input_columns = ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
    output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
    
    # 훈련 데이터 로드
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"훈련 데이터 CSV를 찾을 수 없습니다: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"테스트 데이터 CSV를 찾을 수 없습니다: {test_csv}")
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"훈련 데이터 로드: {train_csv} ({len(train_df)} 샘플)")
    print(f"테스트 데이터 로드: {test_csv} ({len(test_df)} 샘플)")
    
    X_train = train_df[input_columns].to_numpy()
    y_train = train_df[output_columns].to_numpy()
    
    # 데이터 전처리
    X_train_scaled, y_train_scaled = trainer.preprocess_data(X_train, y_train)
    
    # DataLoader 생성
    train_loader = trainer.create_data_loaders(X_train_scaled, y_train_scaled)
    
    # 모델 훈련
    print("\n모델 훈련 시작...")
    start_time = time.time()
    trainer.train(train_loader, epochs=config.get('epochs'))
    training_time = time.time() - start_time
    
    print(f"\n훈련 완료! 소요 시간: {training_time:.2f}초")
    
    # 테스트 데이터 평가
    print("\n테스트 데이터 평가 중...")
    X_test = test_df[input_columns].to_numpy()
    y_test = test_df[output_columns].to_numpy()
    
    X_test_transformed = transform_input_features(X_test)
    X_test_scaled = trainer.feature_scaler.transform(X_test_transformed)
    y_test_transformed = transform_target_outputs(y_test)
    y_test_scaled = trainer.target_scaler.transform(y_test_transformed)
    
    test_dataset = QKDDataset(X_test_scaled, y_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    metrics = trainer.evaluate(test_loader)
    print(f"테스트 MSE (전체): {metrics['overall_mse']:.6e}")
    
    print("\n파라미터별 테스트 오차:")
    for param, stats in metrics['param_errors'].items():
        print(f"  - {param:>4s}: MSE={stats['mse']:.6e}, MAE={stats['mae']:.6e}")
    
    # 모델 저장
    trainer.save_model(output_path)
    
    print("\n" + "=" * 80)
    print(f"훈련 완료! 모델이 {output_path}에 저장되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()
