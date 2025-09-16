#!/usr/bin/env python3
"""
QKD MLP 훈련 스크립트 - 논문 구현
Neural Networks for Parameter Optimization in Quantum Key Distribution
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    def __init__(self, input_size=10, hidden_sizes=[400, 200], output_size=9, dropout_rate=0.1):
        super(QKDMLP, self).__init__()
        
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
    
    def __init__(self, config_path='config/config.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 중인 디바이스: {self.device}")
        
        # 모델 초기화 (논문 구조: 10입력, 9출력, 2은닉층)
        self.model = QKDMLP(input_size=10, hidden_sizes=[400, 200], output_size=9)
        self.model.to(self.device)
        
        # 옵티마이저 및 손실 함수 (논문: Adam, MSE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.MSELoss()  # 논문에서 사용한 MSE Loss
        
        # 스케일러들
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # 훈련 기록
        self.train_losses = []
        self.val_losses = []
        
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
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """데이터 전처리 및 분할 (논문 방식)"""
        print("데이터 전처리 중...")
        
        # 로그 스케일 정규화 (논문에서 언급한 대로)
        # Y_0, eps_sec, eps_cor, N은 로그 스케일로 변환
        log_columns = [2, 7, 8, 9]  # Y_0, eps_sec, eps_cor, N의 인덱스
        
        X_log = X.copy()
        for col in log_columns:
            X_log[:, col] = np.log10(X_log[:, col])
        
        # 입력 데이터 정규화 (논문: 정규화)
        X_scaled = self.feature_scaler.fit_transform(X_log)
        
        # 출력 데이터는 이미 0-1 범위이므로 MinMax 정규화만 적용
        y_scaled = self.target_scaler.fit_transform(y)
        
        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        
        print(f"훈련 데이터: {X_train.shape[0]} 샘플")
        print(f"검증 데이터: {X_val.shape[0]} 샘플")
        
        return X_train, X_val, y_train, y_val
    
    def create_data_loaders(self, X_train, X_val, y_train, y_val, batch_size=64):
        """DataLoader 생성"""
        train_dataset = QKDDataset(X_train, y_train)
        val_dataset = QKDDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
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
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=120, patience=20):
        """모델 훈련 (논문: 120 에포크)"""
        print(f"훈련 시작 - 에포크: {epochs}, 조기 종료: {patience}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="훈련 진행"):
            # 훈련
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증
            val_loss = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                torch.save(self.model.state_dict(), 'best_qkd_mlp_model.pth')
            else:
                patience_counter += 1
            
            # 진행 상황 출력
            if (epoch + 1) % 10 == 0:
                print(f"에포크 {epoch+1}/{epochs} - 훈련 손실: {train_loss:.6f}, 검증 손실: {val_loss:.6f}")
            
            # 조기 종료
            if patience_counter >= patience:
                print(f"조기 종료: {patience} 에포크 동안 개선 없음")
                break
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_qkd_mlp_model.pth'))
        print(f"최고 검증 손실: {best_val_loss:.6f}")
    
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
        
        # MSE 계산
        mse = np.mean((predictions_original - targets_original) ** 2)
        
        # 각 파라미터별 정확도 계산
        param_names = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        param_errors = {}
        
        for i, param_name in enumerate(param_names):
            param_mse = np.mean((predictions_original[:, i] - targets_original[:, i]) ** 2)
            param_mae = np.mean(np.abs(predictions_original[:, i] - targets_original[:, i]))
            param_errors[param_name] = {'mse': param_mse, 'mae': param_mae}
        
        return {
            'overall_mse': mse,
            'param_errors': param_errors,
            'predictions': predictions_original,
            'targets': targets_original
        }
    
    def predict(self, X):
        """새로운 데이터에 대한 예측"""
        self.model.eval()
        
        # 전처리
        X_log = X.copy()
        log_columns = [2, 7, 8, 9]  # Y_0, eps_sec, eps_cor, N
        for col in log_columns:
            X_log[:, col] = np.log10(X_log[:, col])
        
        X_scaled = self.feature_scaler.transform(X_log)
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
                'hidden_sizes': [400, 200],
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
    
    # 훈련기 초기화
    trainer = QKDMLPTrainer()
    
    # 데이터 로드
    try:
        X, y = trainer.load_data('test_qkd_dataset_cleaned.csv')
    except FileNotFoundError:
        print("오류: test_qkd_dataset.csv 파일이 없습니다.")
        print("먼저 data_generator.py를 실행하여 데이터셋을 생성하세요.")
        return
    
    # 데이터 전처리
    X_train, X_val, y_train, y_val = trainer.preprocess_data(X, y)
    
    # DataLoader 생성
    train_loader, val_loader = trainer.create_data_loaders(X_train, X_val, y_train, y_val)
    
    # 모델 훈련
    print("\n모델 훈련 시작...")
    start_time = time.time()
    trainer.train(train_loader, val_loader, epochs=250, patience=250)
    training_time = time.time() - start_time
    
    print(f"\n훈련 완료! 소요 시간: {training_time:.2f}초")
    
    # 모델 평가
    print("\n모델 평가 중...")
    results = trainer.evaluate(val_loader)
    
    print(f"\n전체 MSE: {results['overall_mse']:.6f}")
    print("\n파라미터별 오차:")
    for param, errors in results['param_errors'].items():
        print(f"  {param}: MSE={errors['mse']:.6f}, MAE={errors['mae']:.6f}")
    
    
    # 모델 저장 (PTH 형식)
    trainer.save_model('qkd_mlp_model.pth')
    
    print("\n" + "=" * 80)
    print("훈련 완료! 모델이 qkd_mlp_model.pth로 저장되었습니다.")
    print("=" * 80)

if __name__ == "__main__":
    main()
