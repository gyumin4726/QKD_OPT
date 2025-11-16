import pandas as pd
import numpy as np

def clean_dataset(input_file='./dataset/qkd_dataset_L100.csv', output_file='qkd_dataset_cleaned_L100.csv'):
    """SKR이 -1부터 -10까지의 행들을 제거"""
    
    print(f"데이터셋 로드 중: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"원본 데이터 크기: {len(df)} 행")
    
    # SKR이 -1부터 -10까지인 행들 확인
    invalid_skr_mask = (df['skr'] >= -10) & (df['skr'] <= -1)
    print(f"SKR이 -1부터 -10까지인 행 수: {len(df[invalid_skr_mask])}")
    
    # SKR이 -1부터 -10까지가 아닌 행들만 유지
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
    # 데이터셋 정리
    cleaned_df = clean_dataset()
    
    print("\n데이터셋 정리 완료!")
