import pandas as pd
import numpy as np

def analyze_L_distribution():
    """final_dataset.csv의 L별 데이터 개수 분석"""
    print("=" * 40)
    print("L 값 0~150km 순서대로 전체 데이터 개수")
    print("=" * 40)
    
    try:
        # 데이터 로드
        df = pd.read_csv('final_dataset.csv')
        
        # L 컬럼이 존재하는지 확인
        if 'L' not in df.columns:
            print("오류: 'L' 컬럼이 데이터셋에 없습니다.")
            return None
        
        # L별 데이터 개수 계산
        L_counts = df['L'].value_counts().sort_index()
        unique_L_values = sorted(df['L'].unique())
        
        print(f"{'L(km)':>6} | {'샘플수':>7} | {'비율(%)':>8}")
        print("-" * 25)
        
        # L 값을 0~150 범위로 필터링하고 정렬
        filtered_L_values = [L for L in unique_L_values if 0 <= L <= 150]
        
        for L_val in filtered_L_values:
            count = L_counts[L_val]
            percentage = (count / len(df)) * 100
            print(f"{L_val:6.1f} | {count:7d} | {percentage:8.2f}")
        
        return L_counts
        
    except FileNotFoundError:
        print("오류: final_dataset.csv 파일이 없습니다.")
        print("파일 경로를 확인해주세요.")
        return None
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    # L별 데이터 분포 분석 실행
    analyze_L_distribution()
