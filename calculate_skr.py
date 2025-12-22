"""
SKR(Secure Key Rate) 계산 유틸리티

8개의 QKD 파라미터(mu, nu, vac, p_mu, p_nu, p_vac, p_x, q_x)와 
거리(L)를 입력받아 SKR 값을 계산하는 독립 실행형 스크립트입니다.

- 환경 변수: 파일 상단에서 검출, 광섬유, 오류 정정, 보안, 시스템 파라미터 설정
- 최적화 파라미터: 8개의 QKD 파라미터와 거리 범위 설정
- 메인 실행: 설정된 파라미터로 거리별 SKR 계산 및 결과 출력

사용법:
    1. 파일 상단의 환경 변수 및 파라미터 수정
    2. python calculate_skr.py 실행
"""

import numpy as np
from simulator import skr_simulator

# ============================================================
# 환경 변수 설정
# ============================================================

# 검출 파라미터
ETA_D = 0.045           # 단일 광자 검출기의 검출 효율 (%)
Y_0 = 1.7e-6            # 암계수율 (dark count rate)
E_D = 0.033             # 정렬 오류율 (misalignment rate)

# 광섬유 파라미터
ALPHA = 0.21            # 단일 모드 광섬유의 감쇠 계수

# 오류 정정 파라미터
ZETA = 1.22             # 오류 정정 효율
E_0 = 0.5               # 배경 오류율

# 보안 파라미터
EPS_SEC = 1.0e-10       # 보안 매개변수
EPS_COR = 1.0e-15       # 정확성 매개변수

# 시스템 파라미터
N = 1.0e10              # Alice가 보낸 광 펄스 개수
LAMBDA = None           # Xk에서 관찰된 비트 값 1의 확률

# ============================================================
# 최적화 파라미터 설정
# ============================================================

# QKD 최적화 파라미터 (8개)
MU = 0.521068           # 강도 파라미터 mu
NU = 0.236871           # 강도 파라미터 nu
VAC = 0.034389          # 진공 상태
P_MU = 0.852630         # mu에 대한 확률
P_NU = 0.162988         # nu에 대한 확률
P_VAC = 0.090949        # 진공에 대한 확률
P_X = 0.163874          # X 기저 선택 확률
Q_X = 0.209120          # 수신자의 X 기저 선택 확률

# 거리 범위 설정
L_START = 0             # 시작 거리 (km)
L_END = 131             # 종료 거리 (km, 미포함)
L_STEP = 10             # 거리 간격 (km)

# ============================================================

def calculate_skr(mu, nu, vac, p_mu, p_nu, p_vac, p_x, q_x, L=100):
    # 8개의 파라미터를 배열로 구성
    parameters = np.array([mu, nu, vac, p_mu, p_nu, p_vac, p_x, q_x])
    
    # SKR 계산 (환경 변수를 kwargs로 직접 전달)
    skr = skr_simulator(
        None, parameters, None,
        eta_d=ETA_D, Y_0=Y_0, e_d=E_D,
        alpha=ALPHA, zeta=ZETA, e_0=E_0,
        eps_sec=EPS_SEC, eps_cor=EPS_COR,
        N=N, Lambda=LAMBDA, L=L
    )
    
    return skr


if __name__ == "__main__":
    # 파일 상단의 파라미터로 SKR 계산
    print("=" * 60)
    print("파라미터로 SKR 계산")
    print("=" * 60)
    
    # 상단에서 정의한 파라미터 사용
    mu = MU
    nu = NU
    vac = VAC
    p_mu = P_MU
    p_nu = P_NU
    p_vac = P_VAC
    p_x = P_X
    q_x = Q_X
    
    # 상단에서 정의한 거리 범위 사용
    L_values = np.arange(L_START, L_END, L_STEP)
    
    print(f"\n환경 변수 (Detection, Fiber, Error Correction, Security, System):")
    print(f"  eta_d={ETA_D}, Y_0={Y_0}, e_d={E_D}")
    print(f"  alpha={ALPHA}, zeta={ZETA}, e_0={E_0}")
    print(f"  eps_sec={EPS_SEC}, eps_cor={EPS_COR}, N={N}")
    print(f"\n거리 범위: L={L_START}~{L_END-1} km ({L_STEP} km 간격, 총 {len(L_values)}개)")
    print(f"\n최적화 파라미터 (QKD):")
    print(f"  mu: {mu}")
    print(f"  nu: {nu}")
    print(f"  vac: {vac}")
    print(f"  p_mu: {p_mu}")
    print(f"  p_nu: {p_nu}")
    print(f"  p_vac: {p_vac}")
    print(f"  p_x: {p_x}")
    print(f"  q_x: {q_x}")
    
    print("\n" + "=" * 60)
    print("SKR 계산 중...")
    print("=" * 60)
    
    results = []
    
    try:
        print(f"\n{'L (km)':>8} | {'SKR':>15} | {'상태':>10}")
        print("-" * 40)
        
        for L in L_values:
            skr = calculate_skr(
                mu=mu,
                nu=nu,
                vac=vac,
                p_mu=p_mu,
                p_nu=p_nu,
                p_vac=p_vac,
                p_x=p_x,
                q_x=q_x,
                L=L
            )
            
            results.append({'L': L, 'skr': skr})
            
            if skr < 0:
                status = f"에러({int(skr)})"
                print(f"{L:8.0f} | {skr:15.0f} | {status:>10}")
            else:
                status = "정상"
                print(f"{L:8.0f} | {skr:15.6e} | {status:>10}")
        
        # 요약
        print("\n" + "=" * 60)
        print("요약")
        print("=" * 60)
        
        valid_results = [r for r in results if r['skr'] >= 0]
        error_results = [r for r in results if r['skr'] < 0]
        
        print(f"정상 계산: {len(valid_results)}개")
        print(f"에러 발생: {len(error_results)}개")
        
        if valid_results:
            max_skr = max(valid_results, key=lambda x: x['skr'])
            min_skr = min(valid_results, key=lambda x: x['skr'])
            print(f"\n최대 SKR: {max_skr['skr']:.6e} (L={max_skr['L']} km)")
            print(f"최소 SKR: {min_skr['skr']:.6e} (L={min_skr['L']} km)")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")