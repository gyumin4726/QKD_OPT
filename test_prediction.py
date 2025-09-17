#!/usr/bin/env python3
"""
QKD MLP 모델 예측 테스트 스크립트
"""

import torch
import numpy as np
import pandas as pd
from train import QKDMLPTrainer, set_seed

def evaluate_with_test_data():
    """평가 데이터셋으로 모델 평가 및 성능 측정"""
    print("=" * 60)
    print("QKD MLP 모델 평가")
    print("=" * 60)
    
    # 재현 가능한 결과를 위한 시드 고정
    set_seed(42)
    
    try:
        # 평가 데이터 로드
        df = pd.read_csv('test_data.csv')
        print(f"평가 데이터 로드 완료: {len(df)} 샘플")
        
        input_columns = ['L', 'eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'e_0', 'eps_sec', 'eps_cor', 'N']
        output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
        
        # 훈련기 초기화 (스케일러 로드용)
        trainer = QKDMLPTrainer()
        trainer.load_model('qkd_mlp_model.pth')
        
        # 전체 평가 데이터로 예측
        X_test = df[input_columns].values
        y_test = df[output_columns].values
        predictions = trainer.predict(X_test)
        
        # 전체 성능 계산
        overall_mse = np.mean((predictions - y_test) ** 2)
        print(f"\n전체 MSE: {overall_mse:.6f}")
        
        # 파라미터별 성능 계산
        print("\n파라미터별 오차:")
        param_errors = {}
        for i, param_name in enumerate(output_columns):
            param_mse = np.mean((predictions[:, i] - y_test[:, i]) ** 2)
            param_mae = np.mean(np.abs(predictions[:, i] - y_test[:, i]))
            param_errors[param_name] = {'mse': param_mse, 'mae': param_mae}
            print(f"  {param_name}: MSE={param_mse:.6f}, MAE={param_mae:.6f}")
        
        # 랜덤 5개 샘플의 실제값 vs 예측값 비교
        print("\n" + "=" * 60)
        print("랜덤 5개 샘플 - 실제값 vs 예측값")
        print("=" * 60)
        
        # 랜덤 샘플 인덱스 선택
        random_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
        
        for idx, i in enumerate(random_indices):
            row = df.iloc[i]
            print(f"\n샘플 {idx+1} (인덱스 {i}, L={row['L']:.1f}km):")
            print("  파라미터: 실제값 -> 예측값 (오차)")
            
            for j, col in enumerate(output_columns):
                actual = row[col]
                predicted = predictions[i, j]
                error = abs(actual - predicted)
                
                # 퍼센트 오차 계산 (0으로 나누기 방지)
                if actual != 0:
                    error_percent = (error / abs(actual)) * 100
                else:
                    error_percent = 0 if error == 0 else float('inf')
                
                if col == 'skr':
                    print(f"  {col:>6}: {actual:.2e} -> {predicted:.2e} ({error_percent:.1f}%)")
                else:
                    print(f"  {col:>6}: {actual:.6f} -> {predicted:.6f} ({error_percent:.1f}%)")
        
        # 모든 샘플의 파라미터별 평균 오차 % 계산
        print("\n" + "=" * 60)
        print("전체 테스트 데이터셋 - 파라미터별 평균 오차 %")
        print("=" * 60)
        
        param_avg_errors = {}
        for j, param_name in enumerate(output_columns):
            actual_values = y_test[:, j]
            predicted_values = predictions[:, j]
            
            # 퍼센트 오차 계산: |실제값 - 예측값| / |실제값| * 100
            percent_errors = abs((actual_values - predicted_values) / actual_values) * 100
            avg_error_percent = np.mean(percent_errors)
            
            param_avg_errors[param_name] = avg_error_percent
            print(f"  {param_name:>6}: 평균 {avg_error_percent:.2f}%")
        
        # 전체 평균 오차 % 계산
        overall_avg_error = np.mean(list(param_avg_errors.values()))
        print(f"\n전체 평균 오차 %: {overall_avg_error:.2f}%")
        
        return {
            'overall_mse': overall_mse,
            'param_errors': param_errors,
            'param_avg_error_percent': param_avg_errors,
            'overall_avg_error_percent': overall_avg_error,
            'predictions': predictions,
            'actuals': y_test
        }
        
    except FileNotFoundError:
        print("오류: test_data.csv 파일이 없습니다.")
        print("먼저 data_split.py를 실행하여 데이터를 분할하세요.")
        return None
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    # 평가 데이터로 모델 성능 테스트
    evaluate_with_test_data()
