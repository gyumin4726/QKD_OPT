"""
데이터셋 정리 유틸리티 (파이프라인 2단계)

QKD 데이터셋에서 에러가 발생한 행(SKR 값이 -1부터 -12까지)을 제거하는 스크립트입니다.
SKR 에러 코드를 필터링하여 유효한 데이터만 남긴 후 새로운 CSV 파일로 저장합니다.

파이프라인:
    1. data_generator.py     → raw_dataset_L{L}.csv 생성
    2. clean_dataset.py      → cleaned_dataset_L{L}.csv 생성  ⬅ 현재 파일
    3. data_split.py         → train_L{L}.csv, test_L{L}.csv 생성
    4. train_fttransformer.py → 모델 학습

기능:
    - SKR 값이 -1 ~ -12 범위인 행 제거 (에러 코드)
    - 정리된 데이터셋 저장
    - SKR 분포 통계 출력

사용법:
    1. 파일 상단의 L 값 설정 (data_generator.py와 동일하게)
    2. python clean_dataset.py 실행
"""

import pandas as pd

# ============================================================
# 파일 경로 설정
# ============================================================

L = 100                                                    # 거리 (km)
INPUT_FILE = f'dataset/raw_dataset_L{L}.csv'              # 입력: data_generator.py 출력
OUTPUT_FILE = f'dataset/cleaned_dataset_L{L}.csv'         # 출력: data_split.py 입력

# ============================================================

def clean_dataset(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    print(f"데이터셋 로드 중: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"원본 데이터 크기: {len(df)} 행")
    
    # SKR이 -1부터 -12까지인 행들 확인 (에러 코드)
    invalid_skr_mask = (df['skr'] >= -12) & (df['skr'] <= -1)
    print(f"SKR이 -1부터 -12까지인 행 수: {len(df[invalid_skr_mask])}")
    
    # SKR이 -1부터 -12까지가 아닌 행들만 유지
    df_cleaned = df[~invalid_skr_mask].copy()
    
    print(f"정리된 데이터 크기: {len(df_cleaned)} 행")
    print(f"제거된 행 수: {len(df) - len(df_cleaned)}")
    
    # 정리된 데이터셋 저장
    df_cleaned.to_csv(output_file, index=False)
    print(f"정리된 데이터셋이 {output_file}에 저장되었습니다.")
    
    # SKR 분포 확인
    print(f"\nSKR 분포:")
    print(f"  최소값: {df_cleaned['skr'].min():.2e}")
    print(f"  최대값: {df_cleaned['skr'].max():.2e}")
    print(f"  평균값: {df_cleaned['skr'].mean():.2e}")
    print(f"  중앙값: {df_cleaned['skr'].median():.2e}")
    
    return df_cleaned

if __name__ == "__main__":
    # 상단에서 정의한 경로로 데이터셋 정리
    cleaned_df = clean_dataset()
    
    print("\n데이터셋 정리 완료!")