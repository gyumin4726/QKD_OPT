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
    # vac 값을 변화시키면서 SKR 계산
    print("=" * 60)
    print("vac 변화에 따른 SKR 계산")
    print("=" * 60)
    
    # 고정 파라미터 설정
    L = 100
    mu = 0.399478
    nu = 0.167165
    p_mu = 0.563898
    p_nu = 0.912773
    p_vac = 0.141586
    p_x = 0.332250
    q_x = 0.363639
    
    print(f"\n배경 파라미터: config/config.yaml에서 로드")
    print(f"거리 L: {L} km")
    print(f"\n고정된 파라미터:")
    print(f"  mu: {mu}")
    print(f"  nu: {nu}")
    print(f"  p_mu: {p_mu}")
    print(f"  p_nu: {p_nu}")
    print(f"  p_vac: {p_vac}")
    print(f"  p_x: {p_x}")
    print(f"  q_x: {q_x}")
    
    # vac 범위 설정 (0.0부터 nu보다 작게)
    vac_step = 0.005
    vac_values = np.arange(0.0, nu, vac_step)
    
    print(f"\n" + "=" * 60)
    print(f"vac를 {vac_values[0]:.3f}부터 {vac_values[-1]:.3f}까지 {vac_step} 간격으로 변화")
    print(f"총 {len(vac_values)}개 값 테스트")
    print("=" * 60)
    print(f"\n{'vac':>8} | {'SKR':>15} | {'상태':>10}")
    print("-" * 60)
    
    results = []
    
    try:
        for vac in vac_values:
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
            
            # 결과 저장
            results.append({'vac': vac, 'skr': skr})
            
            # 상태 판단
            if skr < 0:
                status = f"에러({int(skr)})"
            else:
                status = "정상"
            
            # 결과 출력
            if skr < 0:
                print(f"{vac:8.5f} | {skr:15.0f} | {status:>10}")
            else:
                print(f"{vac:8.5f} | {skr:15.6e} | {status:>10}")
        
        # 요약 통계
        print("\n" + "=" * 60)
        print("요약 통계")
        print("=" * 60)
        
        valid_results = [r for r in results if r['skr'] >= 0]
        error_results = [r for r in results if r['skr'] < 0]
        
        print(f"정상 계산: {len(valid_results)}개")
        print(f"에러 발생: {len(error_results)}개")
        
        if valid_results:
            max_skr = max(valid_results, key=lambda x: x['skr'])
            min_skr = min(valid_results, key=lambda x: x['skr'])
            print(f"\n최대 SKR: {max_skr['skr']:.6e} (vac={max_skr['vac']:.5f})")
            print(f"최소 SKR: {min_skr['skr']:.6e} (vac={min_skr['vac']:.5f})")
        
    except FileNotFoundError:
        print("\n오류: config/config.yaml 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")

