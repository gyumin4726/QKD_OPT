import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import (
    QKDDataset,
    QKDMLPTrainer,
    set_seed,
    transform_input_features,
    transform_target_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="L=100 전용 QKD MLP 모델을 테스트 데이터로 평가합니다."
    )
    parser.add_argument(
        "--csv",
        default="dataset/test_L100.csv",
        help="평가에 사용할 L=100 테스트 데이터 CSV 경로 (기본값: dataset/test_L100.csv)",
    )
    parser.add_argument(
        "--model",
        default="qkd_mlp_L100.pth",
        help="평가에 사용할 학습된 모델 경로",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="테스트 DataLoader 배치 크기 (기본값: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 시드 값 (기본값: 42)",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="예측 결과를 CSV로 저장할 경로",
    )
    parser.add_argument(
        "--save-metrics",
        default=None,
        help="평가 지표를 JSON 파일로 저장할 경로",
    )
    return parser.parse_args()


def build_test_loader(trainer, X_test, y_test, batch_size):
    X_transformed = transform_input_features(X_test)
    X_scaled = trainer.feature_scaler.transform(X_transformed)

    y_transformed = transform_target_outputs(y_test)
    y_scaled = trainer.target_scaler.transform(y_transformed)

    dataset = QKDDataset(X_scaled, y_scaled)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def compute_skr_percent_errors(actual_skr, predicted_skr):
    mask = actual_skr != 0
    percent_errors = np.zeros_like(actual_skr)
    percent_errors[mask] = np.abs((actual_skr[mask] - predicted_skr[mask]) / actual_skr[mask]) * 100
    percent_errors[~mask] = np.inf
    return percent_errors


def summarize_percent_errors(percent_errors):
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


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"테스트 데이터셋을 찾을 수 없습니다: {args.csv}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {args.model}")

    set_seed(args.seed)

    input_columns = ["eta_d", "Y_0", "e_d", "alpha", "zeta", "eps_sec", "eps_cor", "N"]
    output_columns = ["mu", "nu", "vac", "p_mu", "p_nu", "p_vac", "p_X", "q_X", "skr"]

    df = pd.read_csv(args.csv)
    print("=" * 90)
    print(f"L=100 테스트 데이터셋 로드 완료: {args.csv}")
    print(f"총 샘플 수: {len(df)}")
    print("=" * 90)

    X_test = df[input_columns].to_numpy()
    y_test = df[output_columns].to_numpy()

    trainer = QKDMLPTrainer()
    trainer.load_model(args.model)

    trainer.config["batch_size"] = args.batch_size
    test_loader = build_test_loader(trainer, X_test, y_test, args.batch_size)

    print("테스트 데이터 평가 중...")
    metrics = trainer.evaluate(test_loader)

    overall_mse = metrics["overall_mse"]
    param_errors = metrics["param_errors"]

    print(f"테스트 MSE (전체): {overall_mse:.6e}")
    print("\n파라미터별 테스트 오차:")
    for param, stats in param_errors.items():
        print(f"  - {param:>4s}: MSE={stats['mse']:.6e}, MAE={stats['mae']:.6e}")

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

    if args.save_predictions:
        output_df = df.copy()
        for i, col in enumerate(output_columns):
            output_df[f"pred_{col}"] = predictions[:, i]

        output_df.to_csv(args.save_predictions, index=False)
        print(f"예측 결과를 '{args.save_predictions}'에 저장했습니다.")

    if args.save_metrics:
        os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
        summary = {
            "overall_mse": overall_mse,
            "param_errors": param_errors,
            "skr_percent_error_summary": skr_summary,
        }
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"평가 지표를 '{args.save_metrics}'에 저장했습니다.")


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 1)
    main()

