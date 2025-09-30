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
        
        # SKR 오차 기준으로 5개 샘플 선택 (최소, 25%, 중간, 75%, 최대)
        print("\n" + "=" * 60)
        print("SKR 오차 분포 및 대표 샘플 분석")
        print("=" * 60)
        
        # SKR 오차 계산 (절대 퍼센트 오차)
        skr_idx = output_columns.index('skr')  # SKR의 인덱스 찾기
        skr_actual = y_test[:, skr_idx]
        skr_predicted = predictions[:, skr_idx]
        skr_percent_errors = np.abs((skr_actual - skr_predicted) / skr_actual) * 100
        
        # SKR 오차 분포 통계
        print(f"SKR 오차 분포 통계:")
        print(f"  최소 오차: {np.min(skr_percent_errors):.2f}%")
        print(f"  1분위수(25%): {np.percentile(skr_percent_errors, 25):.2f}%")
        print(f"  중간값(50%): {np.percentile(skr_percent_errors, 50):.2f}%")
        print(f"  3분위수(75%): {np.percentile(skr_percent_errors, 75):.2f}%")
        print(f"  최대 오차: {np.max(skr_percent_errors):.2f}%")
        print(f"  평균 오차: {np.mean(skr_percent_errors):.2f}%")
        
        print("\n" + "=" * 60)
        print("SKR 오차 기준 대표 샘플 5개 - 실제값 vs 예측값")
        print("=" * 60)
        
        # 오차 기준으로 정렬된 인덱스
        sorted_indices = np.argsort(skr_percent_errors)
        
        # 5개 샘플 인덱스 선택: 최소, 25%, 중간, 75%, 최대
        sample_positions = [0, len(sorted_indices)//4, len(sorted_indices)//2, 
                           3*len(sorted_indices)//4, len(sorted_indices)-1]
        selected_indices = [sorted_indices[pos] for pos in sample_positions]
        sample_labels = ["최소 오차", "1분위수(25%)", "중간값(50%)", "3분위수(75%)", "최대 오차"]
        
        for idx, (i, label) in enumerate(zip(selected_indices, sample_labels)):
            row = df.iloc[i]
            skr_error_percent = skr_percent_errors[i]
            print(f"\n샘플 {idx+1} - {label} (인덱스 {i}, L={row['L']:.1f}km, SKR 오차: {skr_error_percent:.1f}%):")
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
        
        # L별 SKR 평균 오차율 분석
        print("\n" + "=" * 60)
        print("L(거리)별 SKR 평균 오차율 분석")
        print("=" * 60)
        
        # L 값별로 그룹화
        unique_L_values = sorted(df['L'].unique())
        l_skr_errors = {}
        
        for L_val in unique_L_values:
            # 해당 L 값에 대한 데이터 필터링
            L_mask = df['L'] == L_val
            L_indices = np.where(L_mask)[0]
            
            if len(L_indices) > 0:
                # 해당 L에서의 SKR 실제값과 예측값
                L_skr_actual = skr_actual[L_indices]
                L_skr_predicted = skr_predicted[L_indices]
                
                # SKR 퍼센트 오차 계산
                L_skr_percent_errors = np.abs((L_skr_actual - L_skr_predicted) / L_skr_actual) * 100
                
                # 통계 계산
                avg_error = np.mean(L_skr_percent_errors)
                min_error = np.min(L_skr_percent_errors)
                max_error = np.max(L_skr_percent_errors)
                median_error = np.median(L_skr_percent_errors)
                std_error = np.std(L_skr_percent_errors)
                
                l_skr_errors[L_val] = {
                    'count': len(L_indices),
                    'avg_error': avg_error,
                    'min_error': min_error,
                    'max_error': max_error,
                    'median_error': median_error,
                    'std_error': std_error
                }
                
                print(f"L = {L_val:4.1f}km ({len(L_indices):4d}개 샘플):")
                print(f"  평균 오차: {avg_error:6.2f}% | 중간값: {median_error:6.2f}% | 표준편차: {std_error:6.2f}%")
                print(f"  최소 오차: {min_error:6.2f}% | 최대 오차: {max_error:6.2f}%")
        
        # L별 오차율 요약
        print(f"\n{'='*60}")
        print("L별 SKR 오차율 요약 (거리별 순서)")
        print(f"{'='*60}")
        
        # 거리별로 0~150km 순서로 정렬
        sorted_L_errors = sorted(l_skr_errors.items(), key=lambda x: x[0])
        
        print(f"{'L(km)':>6} | {'샘플수':>6} | {'평균오차(%)':>10} | {'중간값(%)':>10} | {'표준편차(%)':>11}")
        print("-" * 60)
        
        for L_val, stats in sorted_L_errors:
            print(f"{L_val:6.1f} | {stats['count']:6d} | {stats['avg_error']:10.2f} | {stats['median_error']:10.2f} | {stats['std_error']:11.2f}")
        
        return {
            'overall_mse': overall_mse,
            'param_errors': param_errors,
            'param_avg_error_percent': param_avg_errors,
            'overall_avg_error_percent': overall_avg_error,
            'l_skr_errors': l_skr_errors,
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
