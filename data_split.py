import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

def split_qkd_dataset(input_file='./dataset/qkd_dataset_cleaned_L20.csv', test_size=0.2, random_state=42):
    """QKD 데이터셋을 훈련/평가용으로 분할"""
    print("=" * 60)
    print("QKD 데이터셋 분할")
    print("=" * 60)
    
    # 데이터 로드
    print(f"데이터 로드 중: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"오류: {input_file} 파일이 없습니다.")
        return
    
    print(f"전체 데이터 크기: {len(df)} 샘플")
    
    # 입력 파일명에서 L 값 추출
    L_match = re.search(r'_L(\d+)', input_file)
    if L_match:
        L_value = L_match.group(1)
    else:
        # L 값이 없으면 데이터에서 추출
        L_value = str(int(df['L'].iloc[0]))
    
    # 입력/출력 컬럼 정의
    input_columns = ['L', 'eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'e_0', 'eps_sec', 'eps_cor', 'N']
    output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
    
    # 데이터 분할 (8:2)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    print(f"훈련 데이터: {len(train_df)} 샘플 ({(1-test_size)*100:.0f}%)")
    print(f"평가 데이터: {len(test_df)} 샘플 ({test_size*100:.0f}%)")
    
    # CSV 파일로 저장 (L 값 포함)
    train_file = f'./dataset/train_L{L_value}.csv'
    test_file = f'./dataset/test_L{L_value}.csv'
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n저장 완료:")
    print(f"  - 훈련 데이터: {train_file}")
    print(f"  - 평가 데이터: {test_file}")
    
    # 데이터 분포 확인
    print(f"\n데이터 분포 확인:")
    print(f"  L 범위 - 훈련: {train_df['L'].min():.1f}~{train_df['L'].max():.1f}km")
    print(f"         평가: {test_df['L'].min():.1f}~{test_df['L'].max():.1f}km")
    print(f"  SKR 범위 - 훈련: {train_df['skr'].min():.2e}~{train_df['skr'].max():.2e}")
    print(f"           평가: {test_df['skr'].min():.2e}~{test_df['skr'].max():.2e}")
    
    return train_df, test_df

if __name__ == "__main__":
    split_qkd_dataset()
