import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from train_fttransformer import (
    QKDDataset,
    FTTransformerTrainer,
    set_seed,
    transform_input_features,
    transform_target_outputs
)

# ============================================
# ===== 여기서 테스트 설정을 변경하세요 =====
# ============================================
MODEL_PATH = "qkd_fttransformer_L100_E200_B128.pth"  # 평가할 모델 파일 경로
TEST_CSV = "dataset/test_L100.csv"  # 테스트 데이터 CSV 파일 경로
TEST_BATCH_SIZE = 512   # 테스트 배치 크기
SHOW_DETAILED = True   # 상세 분석 출력 여부
SHOW_IMPORTANCE = True  # 변수 중요도 분석 출력 여부
# ============================================

def build_test_loader(trainer, X_test, y_test, batch_size):
    """테스트 데이터 로더 생성"""
    X_transformed = transform_input_features(X_test)
    X_scaled = trainer.feature_scaler.transform(X_transformed)

    y_transformed = transform_target_outputs(y_test)
    y_scaled = trainer.target_scaler.transform(y_transformed)

    dataset = QKDDataset(X_scaled, y_scaled)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def compute_skr_percent_errors(actual_skr, predicted_skr):
    """SKR 퍼센트 오차 계산"""
    mask = actual_skr != 0
    percent_errors = np.zeros_like(actual_skr)
    percent_errors[mask] = np.abs((actual_skr[mask] - predicted_skr[mask]) / actual_skr[mask]) * 100
    percent_errors[~mask] = np.inf
    return percent_errors

def summarize_percent_errors(percent_errors):
    """퍼센트 오차 통계 요약"""
    finite_errors = percent_errors[np.isfinite(percent_errors)]
    if finite_errors.size == 0:
        return {
            "min": float("inf"),
            "q1": float("inf"),
            "median": float("inf"),
            "q3": float("inf"),
            "max": float("inf"),
            "mean": float("inf"),
            "std": float("inf"),
        }

    return {
        "min": float(np.min(finite_errors)),
        "q1": float(np.percentile(finite_errors, 25)),
        "median": float(np.percentile(finite_errors, 50)),
        "q3": float(np.percentile(finite_errors, 75)),
        "max": float(np.max(finite_errors)),
        "mean": float(np.mean(finite_errors)),
        "std": float(np.std(finite_errors)),
    }

def print_detailed_analysis(df, predictions, actuals, output_columns):
    """상세 분석 출력"""
    skr_idx = output_columns.index('skr')
    skr_actual = actuals[:, skr_idx]
    skr_predicted = predictions[:, skr_idx]
    skr_percent_errors = np.abs((skr_actual - skr_predicted) / skr_actual) * 100
    
    print("\n" + "=" * 60)
    print("SKR 오차 분포 및 대표 샘플 분석")
    print("=" * 60)
    
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
    
    # 5개 샘플 인덱스 선택
    sample_positions = [0, len(sorted_indices)//4, len(sorted_indices)//2, 
                       3*len(sorted_indices)//4, len(sorted_indices)-1]
    selected_indices = [sorted_indices[pos] for pos in sample_positions]
    sample_labels = ["최소 오차", "1분위수(25%)", "중간값(50%)", "3분위수(75%)", "최대 오차"]
    
    for idx, (i, label) in enumerate(zip(selected_indices, sample_labels)):
        row = df.iloc[i]
        skr_error_percent = skr_percent_errors[i]
        print(f"\n샘플 {idx+1} - {label} (인덱스 {i}, SKR 오차: {skr_error_percent:.1f}%):")
        print("  파라미터: 실제값 -> 예측값 (오차%)")
        
        for j, col in enumerate(output_columns):
            actual = actuals[i, j]
            predicted = predictions[i, j]
            error = abs(actual - predicted)
            
            # 퍼센트 오차 계산
            if actual != 0:
                error_percent = (error / abs(actual)) * 100
            else:
                error_percent = 0 if error == 0 else float('inf')
            
            if col == 'skr':
                print(f"  {col:>6}: {actual:.2e} -> {predicted:.2e} ({error_percent:.1f}%)")
            else:
                print(f"  {col:>6}: {actual:.6f} -> {predicted:.6f} ({error_percent:.1f}%)")
    
    # 파라미터별 평균 오차 % 계산
    print("\n" + "=" * 60)
    print("전체 테스트 데이터셋 - 파라미터별 평균 오차 %")
    print("=" * 60)
    
    param_avg_errors = {}
    for j, param_name in enumerate(output_columns):
        actual_values = actuals[:, j]
        predicted_values = predictions[:, j]
        
        # 퍼센트 오차 계산
        percent_errors = abs((actual_values - predicted_values) / actual_values) * 100
        avg_error_percent = np.mean(percent_errors)
        
        param_avg_errors[param_name] = avg_error_percent
        print(f"  {param_name:>6}: 평균 {avg_error_percent:.2f}%")
    
    # 전체 평균 오차 % 계산
    overall_avg_error = np.mean(list(param_avg_errors.values()))
    print(f"\n전체 평균 오차 %: {overall_avg_error:.2f}%")

def main():
    # 설정값 사용
    model_path = MODEL_PATH
    test_csv = TEST_CSV
    batch_size = TEST_BATCH_SIZE
    seed = 42  # 고정 시드
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"테스트 데이터셋을 찾을 수 없습니다: {test_csv}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    print("=" * 90)
    print(f"QKD FT-Transformer 모델 테스트")
    print(f"모델 파일: {model_path}")
    print(f"테스트 데이터: {test_csv}")
    print("=" * 90)
    
    set_seed(seed)

    input_columns = ["eta_d", "Y_0", "e_d", "alpha", "zeta", "eps_sec", "eps_cor", "N"]
    output_columns = ["mu", "nu", "vac", "p_mu", "p_nu", "p_vac", "p_X", "q_X", "skr"]

    df = pd.read_csv(test_csv)
    print(f"테스트 데이터셋 로드 완료: {test_csv}")
    print(f"총 샘플 수: {len(df)}")

    X_test = df[input_columns].to_numpy()
    y_test = df[output_columns].to_numpy()

    trainer = FTTransformerTrainer()
    trainer.load_model(model_path)

    trainer.config["batch_size"] = batch_size
    test_loader = build_test_loader(trainer, X_test, y_test, batch_size)

    print("\n테스트 데이터 평가 중...")
    metrics = trainer.evaluate(test_loader)

    overall_mse = metrics["overall_mse"]
    param_errors = metrics["param_errors"]

    print(f"\n테스트 MSE (전체): {overall_mse:.6e}")
    print("\n파라미터별 테스트 오차:")
    print(f"{'Parameter':<12} {'MSE':<15} {'MAE':<15}")
    print("-" * 42)
    for param, stats in param_errors.items():
        print(f"{param:<12} {stats['mse']:<15.6e} {stats['mae']:<15.6e}")

    predictions = metrics["predictions"]
    actuals = metrics["targets"]

    skr_idx = output_columns.index("skr")
    skr_percent_errors = compute_skr_percent_errors(actuals[:, skr_idx], predictions[:, skr_idx])
    skr_summary = summarize_percent_errors(skr_percent_errors)

    print("\nSKR 퍼센트 오차 통계:")
    print(
        "  최소: {min:.2f}% | 1분위: {q1:.2f}% | 중간: {median:.2f}% | "
        "3분위: {q3:.2f}% | 최대: {max:.2f}%".format(**skr_summary)
    )
    print(f"  평균: {skr_summary['mean']:.2f}% | 표준편차: {skr_summary['std']:.2f}%")

    # 상세 분석 (옵션)
    if SHOW_DETAILED:
        print_detailed_analysis(df, predictions, actuals, output_columns)

    print("\n" + "=" * 90)
    print("테스트 완료!")
    print("=" * 90)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 1)
    main()

