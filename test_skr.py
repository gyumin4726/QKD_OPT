from simulator import rl_simulator

# 여러 파라미터 조합을 테스트
test_params = [
    # [mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X]
	[0.438556, 0.177481, 0.060121, 0.893263 , 0.921555, 0.151795, 0.479533, 0.154457],

]
for p in test_params:
	skr = rl_simulator(None, p, 0)
	print(f"param: {p}, SKR: {skr}")
