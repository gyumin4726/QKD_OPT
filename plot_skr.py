import matplotlib.pyplot as plt
import numpy as np

# 이미지의 데이터를 정확히 재현
L = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
skr_ga = np.array([3.1e-3, 1.8e-3, 1.1e-3, 6.5e-4, 3.8e-4, 2.2e-4, 1.3e-4, 7.0e-5, 4.0e-5, 2.0e-5, 9.9e-6, 4.3e-6, 1.4e-6, 1.7e-7])
skr_ref = np.array([3e-3, 1.7e-3, 9e-4, 5e-4, 2.9e-4, 1.7e-4, 8e-5, 4e-5, 1.9e-5, 8e-6, 3e-6, 3e-7])

plt.figure(figsize=(10,5))

# 파란색 곡선 (ga)
plt.semilogy(L[:len(skr_ga)], skr_ga, 'o-', color='blue', label='ga', markersize=6)
for x, y in zip(L[:len(skr_ga)], skr_ga):
    plt.text(x, y*1.5, f"{y:.1e}", color='blue', fontsize=7, ha='center')

# 빨간색 곡선 (ref)
plt.semilogy(L[:len(skr_ref)], skr_ref, 's-', color='red', label='ref', markersize=6)
for x, y in zip(L[:len(skr_ref)], skr_ref):
    plt.text(x, y*0.7, f"{y:.1e}", color='red', fontsize=7, ha='center')

# L=100에 추가 포인트 (modified)
plt.semilogy(100, 1.35e-5, 'D', color='green', label='modified', markersize=8)
plt.text(100, 1.35e-5*1.5, f"{1.35e-5:.2e}", color='green', fontsize=7, ha='center')

# vac=0 포인트들
vac0_L = [80, 90, 100, 110]
vac0_skr = [2.035399e-05,8.592851e-06, 2.821089e-06, 3.595696e-07]

for i, (x, y) in enumerate(zip(vac0_L, vac0_skr)):
    if i == 0:
        plt.semilogy(x, y, '^', color='purple', label='vac=0', markersize=8)
    else:
        plt.semilogy(x, y, '^', color='purple', markersize=8)
    
    # 텍스트 위치 조정
    plt.text(x, y*1.5, f"{y:.2e}", color='purple', fontsize=7, ha='center')

plt.xlabel('L')
plt.ylabel('SKR')
plt.title('SKR Comparison')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('skr_comparison.png', dpi=150)
print("그래프가 skr_comparison.png 파일로 저장되었습니다.")
plt.show()
