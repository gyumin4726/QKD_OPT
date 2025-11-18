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
# ======= 여기서 학습 설정을 변경하세요 =======
# ============================================
TRAINING_CONFIG = {
    # 기본 설정
    'L': 100,              # 거리 L (km)
    'epochs': 200,         # 훈련 에포크 수
    'batch_size': 128,      # 배치 크기
    # 최적화 설정
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'dropout_rate': 0.1,
    'loss_scaling': 100,
    # FT-Transformer 전용 설정
    'd_embed': 32,
    'n_heads': 4,
    'n_layers': 3,
    'dim_feedforward': 128
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
    - SKR: log10 변환 (0 혹은 음수 방지를 위해 최소값 고정)
    """
    y_transformed = np.array(y, dtype=np.float64, copy=True)

    if y_transformed.ndim == 1:
        y_transformed = y_transformed.reshape(1, -1)

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

class FTTransformerQKD(nn.Module):
    """QKD 파라미터 최적화를 위한 FT-Transformer"""
    def __init__(self, num_features=8, d_embed=32, n_heads=4, n_layers=3, 
                 dim_feedforward=128, output_size=9, dropout_rate=0.1):
        super(FTTransformerQKD, self).__init__()
        
        self.num_features = num_features
        self.d_embed = d_embed
        
        # 각 스칼라 변수를 임베딩 벡터로 변환
        # 각 변수마다 독립적인 Linear layer 사용 (더 풍부한 표현)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_embed) for _ in range(num_features)
        ])
        
        # Transformer Encoder (positional encoding 없음 - tabular 데이터는 순서 없음)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # Output head (9개 출력: 8개 파라미터 + SKR)
        # 8개 입력 × d_embed = 8 * d_embed 차원 활용
        self.output_head = nn.Sequential(
            nn.Linear(num_features * d_embed, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
        
        # Sigmoid for output range [0, 1]
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_features) - 8개 스칼라 입력
        Returns:
            output: (batch, output_size) - 9개 출력
        """
        batch_size = x.size(0)
        
        # 각 변수를 독립적으로 임베딩
        # x[:, i:i+1]: (batch, 1) → embedding: (batch, d_embed)
        embeddings = []
        for i in range(self.num_features):
            feature_value = x[:, i:i+1]  # (batch, 1)
            embedded = self.feature_embeddings[i](feature_value)  # (batch, d_embed)
            embeddings.append(embedded)
        
        # 모든 임베딩을 쌓아서 sequence로 만듦
        x_embedded = torch.stack(embeddings, dim=1)  # (batch, num_features, d_embed)
        
        # Transformer encoder로 전체 변수 간 상호작용 학습
        x_transformed = self.transformer_encoder(x_embedded)  # (batch, num_features, d_embed)
        
        # Flatten: 8개 입력의 정보를 모두 활용 (8 * d_embed 차원)
        x_flattened = x_transformed.view(batch_size, -1)  # (batch, num_features * d_embed)
        
        # Output head로 최종 예측
        output = self.output_head(x_flattened)  # (batch, output_size)
        
        # Sigmoid로 [0, 1] 범위로 제한
        output = self.sigmoid(output)
        
        return output

class FTTransformerTrainer:
    """FT-Transformer 훈련 및 평가를 위한 클래스"""
    
    def __init__(self, config=None):
        """훈련기 초기화"""
        if config is None:
            config = TRAINING_CONFIG
        self.config = config
        
        self.device = torch.device('cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        # FT-Transformer 모델 초기화
        self.model = FTTransformerQKD(
            num_features=8,
            d_embed=config['d_embed'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            output_size=9,
            dropout_rate=config['dropout_rate']
        )
        self.model.to(self.device)
        
        print(f"모델 구조:")
        print(f"  - 입력 변수: 8개")
        print(f"  - 임베딩 차원: {config['d_embed']}")
        print(f"  - Attention heads: {config['n_heads']}")
        print(f"  - Transformer layers: {config['n_layers']}")
        print(f"  - 출력: 9개 (8 params + SKR)")
        
        # 옵티마이저 및 손실 함수
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        # Learning rate scheduler 추가
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        self.criterion = nn.MSELoss()
        
        # 스케일러들
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        
        # 파라미터별 가중치 설정 (SKR에 높은 가중치)
        self.param_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0]).to(self.device)
        print(f"파라미터 가중치: SKR={self.param_weights[-1]:.1f}x, 나머지={self.param_weights[0]:.1f}x")
        
    def load_data(self, csv_path):
        """CSV 파일에서 데이터 로드"""
        print(f"데이터 로드 중: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 입력 파라미터 (8개)
        input_columns = ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
        
        # 출력 파라미터 (9개)
        output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        
        # 데이터 추출
        X = df[input_columns].values
        y = df[output_columns].values
        
        print(f"데이터 크기: {X.shape[0]} 샘플")
        print(f"입력 차원: {X.shape[1]}, 출력 차원: {y.shape[1]}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """데이터 전처리 (기존과 동일)"""
        print("데이터 전처리 중...")
        
        # 입력 변수 변환
        X_transformed = transform_input_features(X)
        
        print("변환 적용:")
        print("  - Y_0, eps_sec, eps_cor, N: log10(x) 변환")
        print("  - eta_d, e_d, alpha, zeta: 원본 유지 (StandardScaler로 정규화)")
        
        # 입력 데이터 정규화 (StandardScaler)
        X_scaled = self.feature_scaler.fit_transform(X_transformed)
        
        # 출력 데이터 전처리 - SKR에 로그 변환 적용
        y_transformed = transform_target_outputs(y)
        y_scaled = self.target_scaler.fit_transform(y_transformed)
        
        return X_scaled, y_scaled
    
    def weighted_loss(self, output, target):
        """파라미터별 가중치가 적용된 MSE 손실 함수"""
        # 각 파라미터별 MSE 계산
        param_losses = torch.mean((output - target) ** 2, dim=0)  # shape: (9,)
        
        # 가중치 적용
        weighted_losses = param_losses * self.param_weights
        
        # 전체 손실은 가중 평균
        total_loss = torch.sum(weighted_losses) * self.config['loss_scaling']
        
        return total_loss
    
    def create_data_loaders(self, X_train, y_train, batch_size=None):
        """DataLoader 생성"""
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
            generator=generator,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id)
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
            
            # Gradient clipping 추가 (학습 안정화)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def train(self, train_loader, epochs=None):
        """모델 훈련"""
        if epochs is None:
            epochs = self.config['epochs']
        print(f"훈련 시작 - 에포크: {epochs}")
        
        best_train_loss = float('inf')
        
        for epoch in tqdm(range(epochs), desc="훈련 진행"):
            # 훈련
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Learning rate scheduler 업데이트
            self.scheduler.step(train_loss)
            
            # 최고 모델 저장
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                torch.save(self.model.state_dict(), 'best_qkd_fttransformer_model.pth')
            
            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"에포크 {epoch+1}/{epochs} - 훈련 손실: {train_loss:.6f}, LR: {current_lr:.6f}")
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_qkd_fttransformer_model.pth'))
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
        predictions_original[:, -1] = 10 ** predictions_original[:, -1]
        targets_original[:, -1] = 10 ** targets_original[:, -1]
        
        # 평가용 MSE 계산
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
        
        # 전처리
        X_transformed = transform_input_features(X)
        X_scaled = self.feature_scaler.transform(X_transformed)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # 역정규화
        predictions_original = self.target_scaler.inverse_transform(predictions)
        
        # SKR에 대한 역변환
        predictions_original[:, -1] = 10 ** predictions_original[:, -1]
        
        return predictions_original
    
    def save_model(self, path='qkd_fttransformer_model.pth'):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'num_features': 8,
                'd_embed': self.config['d_embed'],
                'n_heads': self.config['n_heads'],
                'n_layers': self.config['n_layers'],
                'dim_feedforward': self.config['dim_feedforward'],
                'output_size': 9,
                'dropout_rate': self.config['dropout_rate']
            }
        }, path)
        print(f"모델이 {path}에 저장되었습니다.")
    
    def load_model(self, path='qkd_fttransformer_model.pth'):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"모델이 {path}에서 로드되었습니다.")
    
    def get_attention_weights(self, X):
        """Attention 가중치 추출 (변수 중요도 분석용)"""
        self.model.eval()
        
        # 전처리
        X_transformed = transform_input_features(X)
        X_scaled = self.feature_scaler.transform(X_transformed)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Forward pass with attention extraction
        # Note: PyTorch TransformerEncoder doesn't expose attention by default
        # This is a simplified version - full implementation would need custom encoder
        print("참고: Attention 가중치 추출은 커스텀 구현이 필요합니다.")
        print("현재는 각 변수의 임베딩 norm으로 중요도를 근사합니다.")
        
        with torch.no_grad():
            batch_size = X_tensor.size(0)
            embeddings = []
            for i in range(8):
                feature_value = X_tensor[:, i:i+1]
                embedded = self.model.feature_embeddings[i](feature_value)
                embeddings.append(embedded)
            
            x_embedded = torch.stack(embeddings, dim=1)
            x_transformed = self.model.transformer_encoder(x_embedded)
            
            # 각 변수의 변환된 임베딩의 norm을 중요도로 사용
            importance = torch.norm(x_transformed, dim=2).mean(dim=0)
            importance = importance / importance.sum()
            
        return importance.cpu().numpy()

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
    output_path = f"qkd_fttransformer_L{L}_E{epochs}_B{batch_size}.pth"
    
    print("=" * 80)
    print(f"QKD FT-Transformer 신경망 훈련 (L={L} km)")
    print("Feature Tokenizer + Transformer for Tabular Data")
    print("=" * 80)
    
    # 설정 사용
    config = TRAINING_CONFIG.copy()
    
    # 재현 가능한 결과를 위한 시드 고정
    set_seed(seed)
    
    # 훈련기 초기화
    trainer = FTTransformerTrainer(config=config)
    
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
    
    # 변수 중요도 분석 (샘플 데이터로)
    print("\n" + "=" * 80)
    print("변수 중요도 분석 (첫 100개 샘플 기준)")
    print("=" * 80)
    sample_importance = trainer.get_attention_weights(X_train[:100])
    var_names = ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
    for i, (name, imp) in enumerate(zip(var_names, sample_importance)):
        print(f"{name:12s}: {imp:.4f}")
    
    print("\n" + "=" * 80)
    print(f"훈련 완료! 모델이 {output_path}에 저장되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()

