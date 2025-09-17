import pandas as pd
import numpy as np
import os

DEFAULT_FILES = {
    'file1': '1.csv',
    'file2': 'test_qkd_dataset_cleaned.csv', 
    'output': 'final_dataset.csv'
}

def merge_qkd_datasets(
    file1=None,
    file2=None, 
    output_file=None,
    remove_invalid_skr=True
):
    """두 QKD 데이터셋을 합치는 함수
    
    Args:
        file1 (str): 첫 번째 데이터셋 파일명 (None이면 기본값 사용)
        file2 (str): 두 번째 데이터셋 파일명 (None이면 기본값 사용)
        output_file (str): 합친 데이터셋 저장 파일명 (None이면 기본값 사용)
        remove_invalid_skr (bool): SKR이 -1인 행들을 제거할지 여부
    """
    
    # 기본값 설정
    file1 = file1 or DEFAULT_FILES['file1']
    file2 = file2 or DEFAULT_FILES['file2']
    output_file = output_file or DEFAULT_FILES['output']
    
    print("=" * 60)
    print("QKD 데이터셋 합치기")
    print("=" * 60)
    
    # 파일 존재 확인
    if not os.path.exists(file1):
        print(f"오류: {file1} 파일이 없습니다.")
        return None
        
    if not os.path.exists(file2):
        print(f"오류: {file2} 파일이 없습니다.")
        return None
    
    # 데이터 로드
    print(f"데이터 로드 중:")
    print(f"  - {file1}")
    df1 = pd.read_csv(file1)
    print(f"    크기: {len(df1)} 행")
    
    print(f"  - {file2}")
    df2 = pd.read_csv(file2)
    print(f"    크기: {len(df2)} 행")
    
    # 컬럼 일치 확인
    if not df1.columns.equals(df2.columns):
        print("경고: 두 데이터셋의 컬럼이 다릅니다!")
        print(f"  {file1} 컬럼: {list(df1.columns)}")
        print(f"  {file2} 컬럼: {list(df2.columns)}")
        return None
    
    # 데이터 합치기
    print(f"\n데이터 합치는 중...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"합친 데이터 크기: {len(merged_df)} 행")
    
    # 중복 제거
    print(f"중복 행 확인 중...")
    original_size = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    duplicates_removed = original_size - len(merged_df)
    
    if duplicates_removed > 0:
        print(f"중복 행 {duplicates_removed}개 제거됨")
        print(f"최종 크기: {len(merged_df)} 행")
    else:
        print("중복 행 없음")
    
    # SKR이 -1인 행들 제거 (선택사항)
    if remove_invalid_skr:
        print(f"\nSKR 값 정리 중...")
        invalid_mask = merged_df['skr'] == -1.0
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"SKR이 -1인 행 {invalid_count}개 발견")
            merged_df = merged_df[~invalid_mask].copy()
            print(f"SKR이 -1인 행들 제거 완료")
            print(f"정리 후 크기: {len(merged_df)} 행")
        else:
            print("SKR이 -1인 행 없음")
    
    # 데이터 분포 확인
    print(f"\n=== 합친 데이터셋 정보 ===")
    print(f"총 샘플 수: {len(merged_df)}")
    print(f"컬럼 수: {len(merged_df.columns)}")
    
    # SKR 분포 확인
    skr_values = merged_df['skr']
    print(f"\nSKR 분포:")
    print(f"  최소값: {skr_values.min():.2e}")
    print(f"  최대값: {skr_values.max():.2e}")
    print(f"  평균값: {skr_values.mean():.2e}")
    print(f"  중앙값: {skr_values.median():.2e}")
    
    # L 값 분포 확인
    L_values = merged_df['L'].value_counts().sort_index()
    print(f"\nL 값 분포:")
    for L, count in L_values.items():
        print(f"  L={L}km: {count}개 샘플")
    
    # 저장
    print(f"\n데이터셋 저장 중: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"저장 완료!")
    
    return merged_df

def main():
    """메인 실행 함수"""
    # 기본 파일명으로 합치기 (파일명은 DEFAULT_FILES에서 자동으로 가져옴)
    merged_df = merge_qkd_datasets(remove_invalid_skr=True)
    
    if merged_df is not None:
        print("\n" + "=" * 60)
        print("데이터셋 합치기 완료!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("데이터셋 합치기 실패!")
        print("=" * 60)

if __name__ == "__main__":
    main()
