#!/usr/bin/env python3
"""
QKD MLP 모델 예측 테스트 스크립트
"""

import torch
import numpy as np
import pandas as pd
from qkd_mlp_train import QKDMLP, QKDMLPTrainer

def test_prediction():
    """훈련된 모델로 예측 테스트"""
    print("=" * 60)
    print("QKD MLP 모델 예측 테스트")
    print("=" * 60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 모델 로드 (weights_only=False로 설정)
        checkpoint = torch.load('qkd_mlp_model.pth', map_location=device, weights_only=False)
        
        # 모델 초기화
        model = QKDMLP(input_size=10, hidden_sizes=[400, 200], output_size=9)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # 스케일러 로드
        feature_scaler = checkpoint['feature_scaler']
        target_scaler = checkpoint['target_scaler']
        
        print("모델 로드 완료!")
        
        # 테스트 데이터 생성 (논문의 예시와 유사)
        test_inputs = np.array([
            [50, 0.045, 1.7e-6, 0.033, 0.21, 1.22, 0.5, 1e-10, 1e-15, 1e10],   # L=50km
            [100, 0.045, 1.7e-6, 0.033, 0.21, 1.22, 0.5, 1e-10, 1e-15, 1e10],  # L=100km
            [110, 0.045, 1.7e-6, 0.033, 0.21, 1.22, 0.5, 1e-10, 1e-15, 1e10],  # L=110km
        ])
        
        input_names = ['L', 'eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'e_0', 'eps_sec', 'eps_cor', 'N']
        output_names = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        
        print("\n테스트 입력 데이터:")
        for i, (name, value) in enumerate(zip(input_names, test_inputs[0])):
            print(f"  {name}: {value}")
        
        # 예측 수행
        print("\n예측 결과:")
        
        # 전처리
        X_log = test_inputs.copy()
        log_columns = [2, 7, 8, 9]  # Y_0, eps_sec, eps_cor, N
        for col in log_columns:
            X_log[:, col] = np.log10(X_log[:, col])
        
        X_scaled = feature_scaler.transform(X_log)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            predictions = model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # 역정규화
        predictions_original = target_scaler.inverse_transform(predictions)
        
        for i, (input_data, prediction) in enumerate(zip(test_inputs, predictions_original)):
            print(f"\nL={input_data[0]}km일 때 예측된 파라미터:")
            for name, value in zip(output_names, prediction):
                if name == 'skr':
                    print(f"  {name}: {value:.2e}")
                else:
                    print(f"  {name}: {value:.6f}")
        
        print("\n예측 완료!")
        
    except FileNotFoundError:
        print("오류: qkd_mlp_model.pth 파일이 없습니다.")
        print("먼저 qkd_mlp_train.py를 실행하여 모델을 훈련하세요.")
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")

def test_with_real_data():
    """실제 데이터셋으로 예측 테스트"""
    print("\n" + "=" * 60)
    print("실제 데이터셋으로 예측 테스트")
    print("=" * 60)
    
    try:
        # 실제 데이터 로드
        df = pd.read_csv('test_qkd_dataset.csv')
        
        # 첫 5개 샘플로 테스트
        test_samples = df.head(5)
        
        input_columns = ['L', 'eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'e_0', 'eps_sec', 'eps_cor', 'N']
        output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        
        # 훈련기 초기화 (스케일러 로드용)
        trainer = QKDMLPTrainer()
        trainer.load_model('qkd_mlp_model.pth')
        
        # 예측 수행
        X_test = test_samples[input_columns].values
        predictions = trainer.predict(X_test)
        
        print("실제 데이터 vs 예측 결과:")
        for i, (_, row) in enumerate(test_samples.iterrows()):
            print(f"\n샘플 {i+1} (L={row['L']}km):")
            print("  실제값 -> 예측값")
            for j, col in enumerate(output_columns):
                actual = row[col]
                predicted = predictions[i, j]
                if col == 'skr':
                    print(f"  {col}: {actual:.2e} -> {predicted:.2e}")
                else:
                    print(f"  {col}: {actual:.6f} -> {predicted:.6f}")
        
    except FileNotFoundError:
        print("오류: test_qkd_dataset.csv 파일이 없습니다.")
    except Exception as e:
        print(f"실제 데이터 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    # 기본 예측 테스트
    test_prediction()
    
    # 실제 데이터로 테스트
    test_with_real_data()
