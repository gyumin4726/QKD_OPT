import numpy as np
from simulator import skr_simulator, QKDSimulatorConfig

def calculate_skr(mu, nu, vac, p_mu, p_nu, p_vac, p_x, q_x, 
                  L=100, config_path='config/config.yaml'):
    """
    8개의 파라미터로 SKR(Secure Key Rate) 계산 (config 파일에서 배경 파라미터 로드)
    
    Args:
        mu: 강도 파라미터 mu
        nu: 강도 파라미터 nu
        vac: 진공 상태 (보통 0으로 고정)
        p_mu: mu에 대한 확률
        p_nu: nu에 대한 확률
        p_vac: 진공에 대한 확률
        p_x: X 기저 선택 확률
        q_x: 수신자의 X 기저 선택 확률
        L: 거리 (km), 기본값 110
        config_path: 설정 파일 경로, 기본값 'config/config.yaml'
    
    Returns:
        float: 계산된 SKR 값
    """
    # 8개의 파라미터를 배열로 구성
    parameters = np.array([mu, nu, vac, p_mu, p_nu, p_vac, p_x, q_x])
    
    # config 파일에서 시스템 설정 로드 (ga_crosscheck.py와 동일한 방식)
    simulator_config = QKDSimulatorConfig.from_yaml(config_path)
    simulator_config.L = L  # L 값만 별도로 설정
    
    # SKR 계산
    skr = skr_simulator(None, parameters, None, simulator_config)
    
    return skr


if __name__ == "__main__":
    # 새로운 파라미터 값으로 L=0~120까지 SKR 계산
    print("=" * 60)
    print("새로운 파라미터로 L=0~120까지 SKR 계산")
    print("=" * 60)
    
    # 파라미터 설정 (사용자 제공 값)
    mu = 0.521068
    nu = 0.236871
    vac = 0.034389
    p_mu = 0.252630
    p_nu = 0.862988
    p_vac = 0.090949
    p_x = 0.163874
    q_x = 0.209120
    
    # L 값 범위 설정 (0부터 120까지 10단위)
    L_values = np.arange(0, 131, 10)
    
    print(f"\n배경 파라미터: config/config.yaml에서 로드")
    print(f"거리 L: 0~120 km (10 km 간격, 총 {len(L_values)}개)")
    print(f"\n파라미터:")
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
        
        # 결과를 배열로 반환
        print("\n" + "=" * 60)
        print("결과 배열 (코드에서 사용)")
        print("=" * 60)
        
        L_values_str = ', '.join(map(str, [r['L'] for r in results]))
        skr_values_str = ', '.join([f"{r['skr']:.6e}" if r['skr'] >= 0 else '0' for r in results])
        
        print(f"L = np.array([{L_values_str}])")
        print(f"skr_modified = np.array([{skr_values_str}])")
        
    except FileNotFoundError:
        print("\n오류: config/config.yaml 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")

