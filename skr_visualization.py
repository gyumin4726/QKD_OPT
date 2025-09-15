import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_config():
    """config.yaml 파일에서 설정을 로드합니다."""
    with open('config/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def create_skr_comparison_with_config():
    """config.yaml의 데이터를 사용하여 SKR 비교 그래프를 생성합니다."""
    
    # config.yaml에서 데이터 로드
    config = load_config()
    ref_skr = config['reference']['skr_values']
    optimized_skr = config['reference']['optimized']
    
    # L 값 (0부터 110까지, 10km 간격)
    L_values = np.arange(0, 120, 10)
    
    # 그래프 설정
    plt.figure(figsize=(12, 8))
    
    # Reference 데이터 플롯
    plt.semilogy(L_values, ref_skr, 'rs-', label='Reference', linewidth=2, markersize=8,
                 markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=1.5)
    
    # Optimized 데이터 플롯 (0이 아닌 값만)
    non_zero_indices = [i for i, val in enumerate(optimized_skr) if val != 0]
    if non_zero_indices:
        L_opt = [L_values[i] for i in non_zero_indices]
        SKR_opt = [optimized_skr[i] for i in non_zero_indices]
        plt.semilogy(L_opt, SKR_opt, 'go-', label='Optimized', linewidth=2, markersize=8,
                     markerfacecolor='green', markeredgecolor='darkgreen', markeredgewidth=1.5)
    
    # 각 데이터 포인트에 SKR 값 라벨 추가
    # Reference 데이터 라벨
    for i, (x, y) in enumerate(zip(L_values, ref_skr)):
        plt.annotate(f'{y:.1e}', (x, y), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=8, color='red')
    
    # Optimized 데이터 라벨 (0이 아닌 값만)
    if non_zero_indices:
        for i, (x, y) in enumerate(zip(L_opt, SKR_opt)):
            plt.annotate(f'{y:.1e}', (x, y), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=8, color='green')
    
    # 그래프 설정
    plt.xlabel('L (km)', fontsize=14, fontweight='bold')
    plt.ylabel('SKR', fontsize=14, fontweight='bold')
    plt.title('SKR Comparison (Config Data)', fontsize=16, fontweight='bold')
    
    # 그리드 설정
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.minorticks_on()
    
    # 범례 설정
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # 축 범위 설정
    plt.xlim(-5, 125)
    plt.ylim(1e-7, 1e-2)
    
    # 축 눈금 설정
    plt.xticks(np.arange(0, 130, 10))
    
    # 레이아웃 조정
    plt.tight_layout()
    
    return plt

def main():
    """메인 함수"""
    print("SKR 비교 시각화를 생성합니다...")
    
    # config.yaml 데이터 기반 그래프 생성
    plt2 = create_skr_comparison_with_config()
    plt2.savefig('skr_comparison_config_data.png', dpi=300, bbox_inches='tight')
    plt2.show()
    
    print("그래프가 성공적으로 생성되었습니다!")
    print("- skr_comparison_config_data.png: config.yaml 데이터 기반 그래프")

if __name__ == "__main__":
    main()
