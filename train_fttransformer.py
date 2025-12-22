"""
FT-Transformer 기반 QKD 파라미터 예측 모델 학습 (파이프라인 4단계)

Feature Tokenizer + Transformer를 사용하여 QKD 환경 변수로부터
최적 파라미터(mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X)와
SKR(Secure Key Rate)을 동시에 예측하는 모델을 학습합니다.

파이프라인:
    1. data_generator.py     → raw_dataset_L{L}.csv 생성
    2. clean_dataset.py      → cleaned_dataset_L{L}.csv 생성
    3. data_split.py         → train_L{L}.csv, test_L{L}.csv 생성
    4. train_fttransformer.py → 모델 학습  ⬅ 현재 파일

주요 기능:
    - FT-Transformer 아키텍처로 tabular 데이터 학습
    - Early stopping 및 learning rate scheduler 지원
    - 데이터 전처리 (로그 변환, 정규화)
    - 모델 저장 및 평가

사용법:
    1. 파일 상단의 L 값 설정 (data_split.py와 동일하게)
    2. python train_fttransformer.py 실행
"""

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

# ============================================================
# 훈련 설정
# ============================================================

# 거리 설정
L = 100                            # 거리 (km)

# 훈련 하이퍼파라미터
EPOCHS = 500                       # 훈련 에포크 수
BATCH_SIZE = 64                    # 배치 크기

# 데이터 설정
INCLUDE_Y_0 = False                # Y_0를 입력 변수로 포함할지 여부

# 데이터 경로 설정
TRAIN_CSV = f"dataset/train_L{L}.csv"                             # 훈련 데이터 경로
TEST_CSV = f"dataset/test_L{L}.csv"                               # 테스트 데이터 경로
OUTPUT_MODEL = f"qkd_fttransformer_L{L}_E{EPOCHS}_B{BATCH_SIZE}.pth"  # 출력 모델 경로

# 데이터 컬럼 정의 (Y_0, e_0, L은 고정값이므로 CSV에 없을 수 있음)
if INCLUDE_Y_0:
    INPUT_COLUMNS = ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
else:
    INPUT_COLUMNS = ['eta_d', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']

OUTPUT_COLUMNS = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
NUM_INPUT_FEATURES = len(INPUT_COLUMNS)

# 재현성 설정
RANDOM_SEED = 42                   # 랜덤 시드

# 옵티마이저 설정
OPTIMIZER = 'Adam'                # 옵티마이저 종류: 'Adam', 'SGD', 'AdamW' 등
LEARNING_RATE = 0.0005            # 학습률
WEIGHT_DECAY = 5e-6               # 가중치 감쇠
DROPOUT_RATE = 0.1                # 드롭아웃 비율
LOSS_SCALING = 1                  # 손실 스케일링

# Learning rate scheduler
SCHEDULER_PATIENCE = 10           # LR 감소 전 대기 에포크
SCHEDULER_FACTOR = 0.5            # LR 감소 비율

# Early stopping
EARLY_STOPPING = True             # Early stopping 사용 여부
EARLY_STOPPING_PATIENCE = 30      # 조기 종료 대기 에포크
EARLY_STOPPING_MIN_DELTA = 1e-6   # 개선 최소 임계값

# FT-Transformer 아키텍처
D_EMBED = 128                     # 임베딩 차원
N_HEADS = 4                       # Attention head 수
N_LAYERS = 4                      # Transformer layer 수
DIM_FEEDFORWARD = 256             # Feedforward 차원

# 디바이스 설정
DEVICE = 'cuda'                   # 'cpu', 'cuda', 'auto'

# ============================================================

# 하위 호환성을 위한 TRAINING_CONFIG
TRAINING_CONFIG = {
    'L': L,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'device': DEVICE,
    'optimizer': OPTIMIZER,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'dropout_rate': DROPOUT_RATE,
    'loss_scaling': LOSS_SCALING,
    'scheduler_patience': SCHEDULER_PATIENCE,
    'scheduler_factor': SCHEDULER_FACTOR,
    'early_stopping': EARLY_STOPPING,
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'early_stopping_min_delta': EARLY_STOPPING_MIN_DELTA,
    'd_embed': D_EMBED,
    'n_heads': N_HEADS,
    'n_layers': N_LAYERS,
    'dim_feedforward': DIM_FEEDFORWARD
}

def set_seed(seed=42):
    print(f"시드 고정: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def transform_input_features(X):
    """
    모델 입력 특성 변환: Y_0 (옵션), eps_sec, eps_cor, N은 로그 변환
    """
    X_transformed = np.array(X, dtype=np.float64, copy=True)

    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(1, -1)

    if INCLUDE_Y_0:
        # Y_0 포함: ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
        # Y_0 (인덱스 1): log10 변환
        X_transformed[:, 1] = np.log10(np.clip(X_transformed[:, 1], a_min=1e-20, a_max=None))
        # eps_sec (인덱스 5): log10 변환
        X_transformed[:, 5] = np.log10(np.clip(X_transformed[:, 5], a_min=1e-30, a_max=None))
        # eps_cor (인덱스 6): log10 변환
        X_transformed[:, 6] = np.log10(np.clip(X_transformed[:, 6], a_min=1e-30, a_max=None))
        # N (인덱스 7): log10 변환
        X_transformed[:, 7] = np.log10(np.clip(X_transformed[:, 7], a_min=1.0, a_max=None))
    else:
        # Y_0 제외: ['eta_d', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
        # eps_sec (인덱스 4): log10 변환
        X_transformed[:, 4] = np.log10(np.clip(X_transformed[:, 4], a_min=1e-30, a_max=None))
        # eps_cor (인덱스 5): log10 변환
        X_transformed[:, 5] = np.log10(np.clip(X_transformed[:, 5], a_min=1e-30, a_max=None))
        # N (인덱스 6): log10 변환
        X_transformed[:, 6] = np.log10(np.clip(X_transformed[:, 6], a_min=1.0, a_max=None))

    return X_transformed

def transform_target_outputs(y):
    """
    모델 출력 변환: p_mu, p_nu, p_vac를 비율로 변환, SKR은 로그 변환
    """
    y_transformed = np.array(y, dtype=np.float64, copy=True)

    if y_transformed.ndim == 1:
        y_transformed = y_transformed.reshape(1, -1)

    # p_mu, p_nu, p_vac를 비율로 변환
    p_mu = y_transformed[:, 3]
    p_nu = y_transformed[:, 4]
    p_vac = y_transformed[:, 5]
    sum_p = p_mu + p_nu + p_vac
    
    y_transformed[:, 3] = p_mu / sum_p
    y_transformed[:, 4] = p_nu / sum_p
    y_transformed[:, 5] = p_vac / sum_p

    # SKR 로그 변환
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
        
        # CLS TOKEN: 학습 가능한 클래스 토큰 (전체 정보를 집약)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_embed))
        
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
        # CLS TOKEN의 d_embed 차원만 사용
        self.output_head = nn.Sequential(
            nn.Linear(d_embed, 128),
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
        
        # CLS TOKEN을 sequence의 첫 번째 위치에 추가
        # cls_token: (1, 1, d_embed) → (batch, 1, d_embed)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_embed)
        x_with_cls = torch.cat([cls_tokens, x_embedded], dim=1)  # (batch, num_features+1, d_embed)
        
        # Transformer encoder로 전체 변수 간 상호작용 학습
        # CLS 토큰이 모든 feature 토큰과 attention을 통해 정보를 집약
        x_transformed = self.transformer_encoder(x_with_cls)  # (batch, num_features+1, d_embed)
        
        # CLS TOKEN만 사용 (첫 번째 위치)
        cls_output = x_transformed[:, 0, :]  # (batch, d_embed)
        
        # Output head로 최종 예측
        output = self.output_head(cls_output)  # (batch, output_size)
        
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
        
        # 디바이스 설정
        device_config = config.get('device', 'auto')
        if device_config == 'auto':
            # GPU 사용 가능하면 GPU, 없으면 CPU 사용
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("경고: CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
                self.device = torch.device('cpu')
        elif device_config == 'cpu':
            self.device = torch.device('cpu')
        else:
            raise ValueError(f"잘못된 device 설정: {device_config}. 'cpu', 'cuda', 또는 'auto'를 사용하세요.")
        
        print(f"사용 중인 디바이스: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # FT-Transformer 모델 초기화
        self.model = FTTransformerQKD(
            num_features=NUM_INPUT_FEATURES,
            d_embed=config['d_embed'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dim_feedforward=config['dim_feedforward'],
            output_size=9,
            dropout_rate=config['dropout_rate']
        )
        self.model.to(self.device)
        
        print(f"모델 구조:")
        print(f"  - 입력 변수: {NUM_INPUT_FEATURES}개 (Y_0 {'포함' if INCLUDE_Y_0 else '제외'})")
        print(f"  - CLS TOKEN: 사용 (전체 정보 집약)")
        print(f"  - 임베딩 차원: {config['d_embed']}")
        print(f"  - Attention heads: {config['n_heads']}")
        print(f"  - Transformer layers: {config['n_layers']}")
        print(f"  - 출력: 9개 (8 params + SKR)")
        
        # 옵티마이저 설정
        optimizer_name = config.get('optimizer', 'Adam')
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config['weight_decay']
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config['weight_decay'],
                momentum=0.9
            )
        elif optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
        # Learning rate scheduler 추가
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=config.get('scheduler_factor', 0.5), 
            patience=config.get('scheduler_patience', 10)
        )
        self.criterion = nn.MSELoss()
        
        # 스케일러들
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        
        # 파라미터별 가중치 설정 (SKR에 높은 가중치)
        self.param_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]).to(self.device)
        print(f"파라미터 가중치: SKR={self.param_weights[-1]:.1f}x, 나머지={self.param_weights[0]:.1f}x")
        
    def load_data(self, csv_path):
        """CSV 파일에서 데이터 로드"""
        print(f"데이터 로드 중: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 상단에 정의된 INPUT_COLUMNS, OUTPUT_COLUMNS 사용
        X = df[INPUT_COLUMNS].values
        y = df[OUTPUT_COLUMNS].values
        
        print(f"데이터 크기: {X.shape[0]} 샘플")
        print(f"입력 차원: {X.shape[1]}, 출력 차원: {y.shape[1]}")
        print(f"Y_0 포함 여부: {INCLUDE_Y_0}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """데이터 전처리"""
        print("데이터 전처리 중...")
        
        # 입력 변수 변환
        X_transformed = transform_input_features(X)
        
        print("입력 변환 적용:")
        print("  - Y_0, eps_sec, eps_cor, N: log10(x) 변환")
        print("  - eta_d, e_d, alpha, zeta: 원본 유지 (MinMaxScaler로 정규화)")
        
        # 입력 데이터 정규화 (MinMaxScaler)
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
    
    def compute_test_loss(self, test_loader):
        """테스트 데이터에 대한 손실 계산 (정규화된 데이터 기준)"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 가중치가 적용된 손실 함수 계산
                loss = self.weighted_loss(output, target)
                total_loss += loss.item()
        
        return total_loss / len(test_loader)
    
    def train(self, train_loader, test_loader=None, epochs=None):
        """모델 훈련
        
        Args:
            train_loader: 훈련 데이터 로더
            test_loader: 테스트 데이터 로더 (None이면 train loss 기준, 있으면 test loss 기준)
            epochs: 훈련 에포크 수
        """
        if epochs is None:
            epochs = self.config['epochs']
        print(f"훈련 시작 - 에포크: {epochs}")
        
        # Early stopping 설정
        use_early_stopping = self.config.get('early_stopping', False)
        early_stopping_patience = self.config.get('early_stopping_patience', 30)
        early_stopping_min_delta = self.config.get('early_stopping_min_delta', 1e-6)
        
        if use_early_stopping:
            print(f"Early stopping 활성화: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        
        if test_loader is not None:
            print("테스트 데이터를 validation set처럼 사용 (test loss 기준 스케줄링 및 early stopping)")
        
        best_test_loss = float('inf') if test_loader is not None else None
        best_train_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        
        for epoch in tqdm(range(epochs), desc="훈련 진행"):
            # 훈련
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 테스트 데이터로 평가 (매 에포크마다)
            if test_loader is not None:
                test_loss = self.compute_test_loss(test_loader)
                self.val_losses.append(test_loss)
                
                # Learning rate scheduler 업데이트 (test loss 기준)
                self.scheduler.step(test_loss)
                
                # 최고 모델 저장 및 개선 확인 (test loss 기준)
                improved = False
                if test_loss < (best_test_loss - early_stopping_min_delta):
                    best_test_loss = test_loss
                    best_epoch = epoch + 1  # best epoch 기록
                    torch.save(self.model.state_dict(), 'best_qkd_fttransformer_model.pth')
                    epochs_without_improvement = 0
                    improved = True
                else:
                    epochs_without_improvement += 1
                
                # 진행 상황 출력
                if (epoch + 1) % 10 == 0 or improved:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    early_stop_info = f", Early stop: {epochs_without_improvement}/{early_stopping_patience}" if use_early_stopping else ""
                    print(f"에포크 {epoch+1}/{epochs} - 훈련 손실: {train_loss:.6f}, 테스트 손실: {test_loss:.6f}, LR: {current_lr:.6f}{early_stop_info}")
                
                # Early stopping 체크
                if use_early_stopping and epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping 발동! {early_stopping_patience} 에포크 동안 개선이 없었습니다.")
                    print(f"최고 테스트 손실: {best_test_loss:.6f} (에포크 {best_epoch})")
                    if best_epoch < (epoch + 1):
                        print(f"→ {epoch + 1 - best_epoch} 에포크 전의 체크포인트로 복원됩니다.")
                    break
            else:
                # 테스트 데이터가 없으면 기존 방식 (train loss 기준)
                self.scheduler.step(train_loss)
                
                # 최고 모델 저장 및 개선 확인
                improved = False
                if train_loss < (best_train_loss - early_stopping_min_delta):
                    best_train_loss = train_loss
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), 'best_qkd_fttransformer_model.pth')
                    epochs_without_improvement = 0
                    improved = True
                else:
                    epochs_without_improvement += 1
                
                # 진행 상황 출력
                if (epoch + 1) % 10 == 0 or improved:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    early_stop_info = f", Early stop: {epochs_without_improvement}/{early_stopping_patience}" if use_early_stopping else ""
                    print(f"에포크 {epoch+1}/{epochs} - 훈련 손실: {train_loss:.6f}, LR: {current_lr:.6f}{early_stop_info}")
                
                # Early stopping 체크
                if use_early_stopping and epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping 발동! {early_stopping_patience} 에포크 동안 개선이 없었습니다.")
                    print(f"최고 훈련 손실: {best_train_loss:.6f} (에포크 {best_epoch})")
                    if best_epoch < (epoch + 1):
                        print(f"→ {epoch + 1 - best_epoch} 에포크 전의 체크포인트로 복원됩니다.")
                    break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_qkd_fttransformer_model.pth'))
        if test_loader is not None:
            print(f"\n최고 테스트 손실: {best_test_loss:.6f} (에포크 {best_epoch})")
        else:
            print(f"\n최고 훈련 손실: {best_train_loss:.6f} (에포크 {best_epoch})")
        print(f"총 훈련 에포크: {epoch + 1}/{epochs}")
    
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
        
        # SKR에 대한 역변환 (log10 -> 원본 스케일)
        predictions_original[:, -1] = 10 ** predictions_original[:, -1]
        
        # 비율(r_mu, r_nu, r_vac) 재정규화 (합=1 보장)
        r_mu = predictions_original[:, 3]
        r_nu = predictions_original[:, 4]
        r_vac = predictions_original[:, 5]
        sum_r = r_mu + r_nu + r_vac
        predictions_original[:, 3] = r_mu / sum_r
        predictions_original[:, 4] = r_nu / sum_r
        predictions_original[:, 5] = r_vac / sum_r
        
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
    
def main():
    print("=" * 80)
    print(f"QKD FT-Transformer 신경망 훈련 (L={L} km)")
    print("Feature Tokenizer + Transformer for Tabular Data")
    print("=" * 80)
    
    config = TRAINING_CONFIG.copy()
    
    set_seed(RANDOM_SEED)
    
    trainer = FTTransformerTrainer(config=config)
    
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"훈련 데이터 CSV를 찾을 수 없습니다: {TRAIN_CSV}")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"테스트 데이터 CSV를 찾을 수 없습니다: {TEST_CSV}")
    
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"훈련 데이터 로드: {TRAIN_CSV} ({len(train_df)} 샘플)")
    print(f"테스트 데이터 로드: {TEST_CSV} ({len(test_df)} 샘플)")
    
    X_train = train_df[INPUT_COLUMNS].to_numpy()
    y_train = train_df[OUTPUT_COLUMNS].to_numpy()
    
    X_train_scaled, y_train_scaled = trainer.preprocess_data(X_train, y_train)
    
    X_test = test_df[INPUT_COLUMNS].to_numpy()
    y_test = test_df[OUTPUT_COLUMNS].to_numpy()
    
    X_test_transformed = transform_input_features(X_test)
    X_test_scaled = trainer.feature_scaler.transform(X_test_transformed)
    y_test_transformed = transform_target_outputs(y_test)
    y_test_scaled = trainer.target_scaler.transform(y_test_transformed)
    
    # DataLoader 생성
    train_loader = trainer.create_data_loaders(X_train_scaled, y_train_scaled)
    test_dataset = QKDDataset(X_test_scaled, y_test_scaled)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 모델 훈련 (테스트 데이터를 validation set처럼 사용)
    print("\n모델 훈련 시작...")
    start_time = time.time()
    trainer.train(train_loader, test_loader=test_loader, epochs=config.get('epochs'))
    training_time = time.time() - start_time
    
    print(f"\n훈련 완료! 소요 시간: {training_time:.2f}초")
    
    # 최종 테스트 데이터 평가
    print("\n최종 테스트 데이터 평가 중...")
    
    metrics = trainer.evaluate(test_loader)
    print(f"테스트 MSE (전체): {metrics['overall_mse']:.6e}")
    
    print("\n파라미터별 테스트 오차:")
    for param, stats in metrics['param_errors'].items():
        print(f"  - {param:>4s}: MSE={stats['mse']:.6e}, MAE={stats['mae']:.6e}")
    
    # 모델 저장
    trainer.save_model(OUTPUT_MODEL)
    
    print("\n" + "=" * 80)
    print(f"훈련 완료! 모델이 {OUTPUT_MODEL}에 저장되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()