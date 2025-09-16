from simulator import skr_simulator
import simulator
import numpy as np
import yaml

# ga_crosscheck.py와 정확히 동일한 방식으로 설정
with open('config/config_crosscheck.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 상수 정의 (YAML에서 로드)
simulator.eta_d = float(config['detection']['eta_d'])
simulator.Y_0 = float(config['detection']['Y_0'])
simulator.e_d = float(config['detection']['e_d'])
simulator.alpha = float(config['fiber']['alpha'])
simulator.zeta = float(config['error_correction']['zeta'])
simulator.e_0 = float(config['error_correction']['e_0'])
simulator.eps_sec = float(config['security']['eps_sec'])
simulator.eps_cor = float(config['security']['eps_cor'])
simulator.N = float(config['system']['N'])

# 파생 상수
simulator.eps = simulator.eps_sec/23
simulator.beta = np.log(1/simulator.eps)

# L 설정
simulator.L = 20

print("=== ga_crosscheck.py와 동일한 설정 ===")
print(f"L: {simulator.L}")
print(f"eta_d: {simulator.eta_d}")
print(f"Y_0: {simulator.Y_0}")
print(f"e_d: {simulator.e_d}")
print(f"alpha: {simulator.alpha}")
print(f"zeta: {simulator.zeta}")
print(f"e_0: {simulator.e_0}")
print(f"eps_sec: {simulator.eps_sec}")
print(f"eps_cor: {simulator.eps_cor}")
print(f"N: {simulator.N}")
print(f"eps: {simulator.eps}")
print(f"beta: {simulator.beta}")

# ga_crosscheck.py에서 출력된 정확한 파라미터 사용
hyperparams = [0.530750, 0.240146, 0.077156, 0.052039, 0.859464, 0.034550, 0.303349, 0.047854]
print(f"\n하이퍼파라미터: {hyperparams}")

# SKR 계산
skr_value = skr_simulator(None, hyperparams, 0)
print(f"\nSKR: {skr_value:.6e}")

# ga_crosscheck.py 결과와 비교
print(f"ga_crosscheck.py 결과: 9.663955e-04")
print(f"차이: {abs(skr_value - 9.663955e-04):.2e}")
