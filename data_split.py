"""
데이터셋 분할 유틸리티 (파이프라인 3단계)

QKD 데이터셋을 훈련 데이터와 평가 데이터로 분할하는 스크립트입니다.

파이프라인:
    1. data_generator.py     → raw_dataset_L{L}.csv 생성
    2. clean_dataset.py      → cleaned_dataset_L{L}.csv 생성
    3. data_split.py         → train_L{L}.csv, test_L{L}.csv 생성  ⬅ 현재 파일
    4. train_fttransformer.py → 모델 학습

기능:
    - CSV 데이터셋을 train/test로 분할
    - 분할된 데이터를 별도 CSV로 저장
    - 데이터 분포 통계 출력

사용법:
    1. 파일 상단의 L 값 설정 (clean_dataset.py와 동일하게)
    2. python data_split.py 실행
"""

import pandas as pd
import re
from sklearn.model_selection import train_test_split

# ============================================================
# 분할 설정
# ============================================================

L = 100                                                  # 거리 (km)
INPUT_FILE = f'dataset/cleaned_dataset_L{L}.csv'        # 입력: clean_dataset.py 출력
OUTPUT_DIR = 'dataset'                                   # 출력 디렉토리 (train_L{L}.csv, test_L{L}.csv)
TEST_SIZE = 0.2                                          # 평가 데이터 비율 (0.2 = 20%)
RANDOM_STATE = 42                                        # 재현성을 위한 랜덤 시드

# ============================================================

def split_qkd_dataset(input_file=INPUT_FILE, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    print("=" * 60)
    print("QKD 데이터셋 분할")
    print("=" * 60)
    
    print(f"데이터 로드 중: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: {input_file} 파일이 없습니다.")
        return
    
    print(f"전체 데이터 크기: {len(df)} 샘플")
    
    # 파일명에서 L 값 추출
    L_match = re.search(r'_L(\d+)', input_file)
    if L_match:
        L_value = L_match.group(1)
    else:
        L_value = "unknown"
    
    # 데이터 분할
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"훈련 데이터: {len(train_df)} 샘플 ({(1-test_size)*100:.0f}%)")
    print(f"평가 데이터: {len(test_df)} 샘플 ({test_size*100:.0f}%)")
    
    train_file = f'{OUTPUT_DIR}/train_L{L_value}.csv'
    test_file = f'{OUTPUT_DIR}/test_L{L_value}.csv'
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n저장 완료:")
    print(f"  - 훈련 데이터: {train_file}")
    print(f"  - 평가 데이터: {test_file}")
    
    print(f"\n데이터 분포:")
    print(f"  SKR 범위 - 훈련: {train_df['skr'].min():.2e}~{train_df['skr'].max():.2e}")
    print(f"           평가: {test_df['skr'].min():.2e}~{test_df['skr'].max():.2e}")
    
    return train_df, test_df

if __name__ == "__main__":
    split_qkd_dataset()