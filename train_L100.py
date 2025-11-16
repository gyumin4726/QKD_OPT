import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train import (
    TRAINING_CONFIG,
    QKDDataset,
    QKDMLPTrainer,
    set_seed,
    transform_input_features,
    transform_target_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="cleaned L=100 데이터셋을 이용해 QKD MLP 모델을 학습합니다."
    )
    parser.add_argument(
        "--split-source",
        default="dataset/qkd_dataset_cleaned_L100.csv",
        help="train/test 분할에 사용할 원본 L=100 데이터셋 경로 (기본값: dataset/qkd_dataset_cleaned_L100.csv)",
    )
    parser.add_argument(
        "--train-csv",
        default="dataset/train_L100.csv",
        help="학습에 사용할 훈련 데이터 CSV 경로 (기본값: dataset/train_L100.csv)",
    )
    parser.add_argument(
        "--test-csv",
        default="dataset/test_L100.csv",
        help="평가에 사용할 테스트 데이터 CSV 경로 (기본값: dataset/test_L100.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="train/test 분할 비율 (기본값: 0.2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="훈련 에포크 수. 지정하지 않으면 기본 설정을 사용합니다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="배치 크기. 지정하지 않으면 기본 설정을 사용합니다.",
    )
    parser.add_argument(
        "--output",
        default="qkd_mlp_L100.pth",
        help="학습된 모델을 저장할 경로",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 시드 값 (기본값: 42)",
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="train/test CSV가 이미 존재해도 강제로 분할을 다시 수행합니다.",
    )
    return parser.parse_args()

def prepare_train_loader(trainer, X_train, y_train):
    X_train_scaled, y_train_scaled = trainer.preprocess_data(X_train, y_train)
    train_loader = trainer.create_data_loaders(X_train_scaled, y_train_scaled)

    return train_loader


def prepare_test_loader(trainer, X_test, y_test):
    X_test_transformed = transform_input_features(X_test)
    X_test_scaled = trainer.feature_scaler.transform(X_test_transformed)
    y_test_transformed = transform_target_outputs(y_test)
    y_test_scaled = trainer.target_scaler.transform(y_test_transformed)

    test_dataset = QKDDataset(X_test_scaled, y_test_scaled)
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer.config["batch_size"],
        shuffle=False,
    )

    return test_loader


def split_dataset_if_needed(args, input_columns, output_columns):
    train_exists = os.path.exists(args.train_csv)
    test_exists = os.path.exists(args.test_csv)

    split_source = args.split_source
    if split_source and isinstance(split_source, str) and split_source.lower() == "none":
        split_source = None

    if args.force_split or not (train_exists and test_exists):
        if split_source is None:
            raise ValueError("train/test 분할을 수행하려면 --split-source 경로를 지정해야 합니다.")
        if not os.path.exists(split_source):
            raise FileNotFoundError(f"분할 원본 데이터셋을 찾을 수 없습니다: {split_source}")

        print("=" * 90)
        print(f"L=100 원본 데이터셋 로드 중: {split_source}")
        df = pd.read_csv(split_source)
        print(f"총 샘플 수: {len(df)}")
        print("=" * 90)

        X = df[input_columns].to_numpy()
        y = df[output_columns].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
            shuffle=True,
        )

        train_df = pd.DataFrame(np.hstack([X_train, y_train]), columns=input_columns + output_columns)
        test_df = pd.DataFrame(np.hstack([X_test, y_test]), columns=input_columns + output_columns)

        train_dir = os.path.dirname(args.train_csv) or "."
        test_dir = os.path.dirname(args.test_csv) or "."
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_df.to_csv(args.train_csv, index=False)
        test_df.to_csv(args.test_csv, index=False)

        print(
            f"train/test 분할 완료 - train: {args.train_csv} ({len(train_df)} 샘플), "
            f"test: {args.test_csv} ({len(test_df)} 샘플)"
        )
    else:
        print(f"기존 train/test CSV를 사용합니다 - train: {args.train_csv}, test: {args.test_csv}")


def main():
    args = parse_args()

    config = TRAINING_CONFIG.copy()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    set_seed(args.seed)

    trainer = QKDMLPTrainer(config=config)

    input_columns = ["eta_d", "Y_0", "e_d", "alpha", "zeta", "eps_sec", "eps_cor", "N"]
    output_columns = ["mu", "nu", "vac", "p_mu", "p_nu", "p_vac", "p_X", "q_X", "skr"]

    split_dataset_if_needed(args, input_columns, output_columns)

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"훈련 데이터 CSV를 찾을 수 없습니다: {args.train_csv}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"테스트 데이터 CSV를 찾을 수 없습니다: {args.test_csv}")

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    print(f"훈련 데이터 로드: {args.train_csv} ({len(train_df)} 샘플)")
    print(f"테스트 데이터 로드: {args.test_csv} ({len(test_df)} 샘플)")

    X_train = train_df[input_columns].to_numpy()
    y_train = train_df[output_columns].to_numpy()

    train_loader = prepare_train_loader(trainer, X_train, y_train)

    print("\n모델 훈련 시작...")
    start_time = time.time()
    trainer.train(train_loader, epochs=config.get("epochs"))
    elapsed = time.time() - start_time
    print(f"훈련 소요 시간: {elapsed:.2f}초")

    print("\n테스트 데이터 평가 중...")
    X_test = test_df[input_columns].to_numpy()
    y_test = test_df[output_columns].to_numpy()
    test_loader = prepare_test_loader(trainer, X_test, y_test)
    metrics = trainer.evaluate(test_loader)
    print(f"테스트 MSE (전체): {metrics['overall_mse']:.6e}")

    print("\n파라미터별 테스트 오차:")
    for param, stats in metrics["param_errors"].items():
        print(f"  - {param:>4s}: MSE={stats['mse']:.6e}, MAE={stats['mae']:.6e}")

    trainer.save_model(args.output)
    print(f"학습된 모델이 '{args.output}'에 저장되었습니다.")


if __name__ == "__main__":
    main()

