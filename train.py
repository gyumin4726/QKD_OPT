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

TRAINING_CONFIG = {
    'epochs': 120,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'dropout_rate': 0.1,
    'loss_scaling': 100
}

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
        self.model = QKDMLP(input_size=10, hidden_sizes=[512, 256], output_size=9, 
                           dropout_rate=config['dropout_rate'])
        self.model.to(self.device)
        
        # 옵티마이저 및 손실 함수 (설정값 사용)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=config['learning_rate'], 
                                   weight_decay=config['weight_decay'])
        self.criterion = nn.MSELoss()  # 논문에서 사용한 MSE Loss
        
        # 스케일러들
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
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
        
        # 입력 파라미터 (10개)
        input_columns = ['L', 'eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'e_0', 'eps_sec', 'eps_cor', 'N']
        
        # 출력 파라미터 (9개: 8개 파라미터 + SKR)
        output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        
        # 데이터 추출
        X = df[input_columns].values
        y = df[output_columns].values
        
        print(f"데이터 크기: {X.shape[0]} 샘플")
        print(f"입력 차원: {X.shape[1]}, 출력 차원: {y.shape[1]}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """개선된 데이터 전처리 - 모든 변수에 적절한 변환 적용"""
        print("개선된 데이터 전처리 중...")
        
        # 입력 변수별 적절한 변환 적용
        # ['L', 'eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'e_0', 'eps_sec', 'eps_cor', 'N']
        X_transformed = X.copy()
        
        # 1. L (0~150): 0값 때문에 log1p 사용 (log(1+x))
        X_transformed[:, 0] = np.log1p(X_transformed[:, 0])  # L
        
        # 2. eta_d (0.02~0.08): 100배 스케일링으로 2~8 범위로 변환
        X_transformed[:, 1] = X_transformed[:, 1] * 100  # eta_d
        
        # 3. Y_0 (1e-7~1e-5): 로그 변환 필요
        X_transformed[:, 2] = np.log10(X_transformed[:, 2])  # Y_0
        
        # 4. e_d (0.02~0.05): 100배 스케일링으로 2~5 범위로 변환
        X_transformed[:, 3] = X_transformed[:, 3] * 100  # e_d
        
        # 5. alpha (0.18~0.24): 100배 스케일링으로 18~24 범위로 변환
        X_transformed[:, 4] = X_transformed[:, 4] * 100  # alpha
        
        # 6. zeta (1.1~1.4): 10배 스케일링으로 11~14 범위로 변환
        X_transformed[:, 5] = X_transformed[:, 5] * 10  # zeta
        
        # 7. e_0 (0.4~0.6): 10배 스케일링으로 4~6 범위로 변환
        X_transformed[:, 6] = X_transformed[:, 6] * 10  # e_0
        
        # 8. eps_sec (1e-12~1e-8): 로그 변환 필요
        X_transformed[:, 7] = np.log10(X_transformed[:, 7])  # eps_sec
        
        # 9. eps_cor (1e-18~1e-13): 로그 변환 필요
        X_transformed[:, 8] = np.log10(X_transformed[:, 8])  # eps_cor
        
        # 10. N (1e9~1e11): 로그 변환 필요
        X_transformed[:, 9] = np.log10(X_transformed[:, 9])  # N
        
        print("변환 적용:")
        print("  - L: log1p(x) 변환 (0값 처리)")
        print("  - Y_0, eps_sec, eps_cor, N: log10(x) 변환")
        print("  - eta_d, e_d, alpha: 100배 스케일링")
        print("  - zeta, e_0: 10배 스케일링")
        
        # 입력 데이터 정규화 (StandardScaler)
        X_scaled = self.feature_scaler.fit_transform(X_transformed)
        
        # 출력 데이터는 이미 0-1 범위이므로 MinMax 정규화만 적용
        y_scaled = self.target_scaler.fit_transform(y)
        
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
        X_transformed = X.copy()
        
        # 동일한 변환 적용
        X_transformed[:, 0] = np.log1p(X_transformed[:, 0])    # L: log1p
        X_transformed[:, 1] = X_transformed[:, 1] * 100        # eta_d: 100배
        X_transformed[:, 2] = np.log10(X_transformed[:, 2])    # Y_0: log10
        X_transformed[:, 3] = X_transformed[:, 3] * 100        # e_d: 100배
        X_transformed[:, 4] = X_transformed[:, 4] * 100        # alpha: 100배
        X_transformed[:, 5] = X_transformed[:, 5] * 10         # zeta: 10배
        X_transformed[:, 6] = X_transformed[:, 6] * 10         # e_0: 10배
        X_transformed[:, 7] = np.log10(X_transformed[:, 7])    # eps_sec: log10
        X_transformed[:, 8] = np.log10(X_transformed[:, 8])    # eps_cor: log10
        X_transformed[:, 9] = np.log10(X_transformed[:, 9])    # N: log10
        
        X_scaled = self.feature_scaler.transform(X_transformed)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # 역정규화
        predictions_original = self.target_scaler.inverse_transform(predictions)
        
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
                'input_size': 10,
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
    print("=" * 80)
    print("QKD MLP 신경망 훈련 - 논문 구현")
    print("Neural Networks for Parameter Optimization in Quantum Key Distribution")
    print("=" * 80)
    
    # 재현 가능한 결과를 위한 시드 고정
    set_seed(42)
    
    # 훈련기 초기화
    trainer = QKDMLPTrainer()
    
    # 훈련 데이터 로드
    try:
        X_train, y_train = trainer.load_data('train_data.csv')
    except FileNotFoundError:
        print("오류: train_data.csv 파일이 없습니다.")
        print("먼저 data_split.py를 실행하여 데이터를 분할하세요.")
        return
    
    # 데이터 전처리 (정규화만)
    X_train_scaled, y_train_scaled = trainer.preprocess_data(X_train, y_train)
    
    # DataLoader 생성
    train_loader = trainer.create_data_loaders(X_train_scaled, y_train_scaled)
    
    # 모델 훈련 (설정값 사용)
    print("\n모델 훈련 시작...")
    start_time = time.time()
    trainer.train(train_loader)  # epochs는 설정값에서 자동으로 가져옴
    training_time = time.time() - start_time
    
    print(f"\n훈련 완료! 소요 시간: {training_time:.2f}초")
    
    # 모델 저장 (PTH 형식)
    trainer.save_model('qkd_mlp_model.pth')
    
    print("\n" + "=" * 80)
    print("훈련 완료! 모델이 qkd_mlp_model.pth로 저장되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()
