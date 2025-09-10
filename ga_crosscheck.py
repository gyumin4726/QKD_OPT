import numpy as np
import pandas as pd
import pygad
import yaml

import warnings
warnings.filterwarnings('ignore')

# RL 시뮬레이터 import
from simulator import rl_simulator, normalize_p

# 설정 파일 로드
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 상수 정의 (YAML에서 로드)
eta_d = float(config['detection']['eta_d'])
Y_0 = float(config['detection']['Y_0'])
e_d = float(config['detection']['e_d'])
alpha = float(config['fiber']['alpha'])
zeta = float(config['error_correction']['zeta'])
e_0 = float(config['error_correction']['e_0'])
eps_sec = float(config['security']['eps_sec'])
eps_cor = float(config['security']['eps_cor'])
N = float(config['system']['N'])
Lambda = config['system']['Lambda']  # None일 수 있음

# 파생 상수
eps = eps_sec/23
beta = np.log(1/eps)

# L은 직접 설정
L = 100

# simulator.py의 L 값 업데이트
import simulator
simulator.L = L

# 재현 가능한 결과를 위한 시드 설정
import random
random.seed(42)
np.random.seed(42)

def define_ga(co_type, mu_type, sel_type, 
              gen = 200,
              num_parents_mating = 60, sol_per_pop = 200, keep_parents = 50, keep_elitism = 10, K_tournament = 8, crossover_probability = 0.8, mutation_probability = 0.02, mutation_percent_genes = "default",
              random_seed = 42):
    """유전 알고리즘 인스턴스를 정의하는 함수"""
    
    ga_instance = pygad.GA(num_generations = gen,   #(논문 : 최대 1000)                    # 세대 수
                    num_parents_mating = num_parents_mating,   #(논문 : 30)               # 부모로 선택될 솔루션의 수

                    fitness_func = rl_simulator,
                    fitness_batch_size = None,                                           # 배치 단위로 적합도 함수를 계산, 적합도 함수는 각 배치에 대해 한 번씩 호출

                    initial_population = None,                                           # 사용자 정의 초기 개체군, num_genes와 크기가 같아야 함
                    sol_per_pop = sol_per_pop,                                           # 한 세대에 포함되는 솔루션(염색체)의 수, 크면 탐색 다양성이 높아짐, 작으면 빠르게 수렴하지만 최적해를 놓칠 수 있음, initial population이 있으면 작동하지 않음
                    num_genes = 8,                                                       # 염색체 내 유전자 수, initial_population을 사용하는 경우 이 매개변수가 필요하지 않음
                    gene_type = [float, 6],                                              # 유전자 유형, 각 개별 유전자의 데이터 유형 및 소수점도 지정 가능, 리스트 형식 e.g. [int, float, bool, int]

                    init_range_low = 0,                                                  # 초기 모집단의 유전자 값이 선택되는 임의 범위의 하한, initial_population이 있으면 필요 없음
                    init_range_high = 1,                                                 # 초기 모집단의 유전자 값이 선택되는 임의 범위의 상한,

                    parent_selection_type = sel_type,                                    # 부모 선택 유형, sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection)
                    keep_parents = keep_parents,                                         # 현재 개체군에 유지할 부모의 수, -1 : 모든 부모를 개체군에 유지, keep_elitism이 0인 경우에만 작동
                    keep_elitism = keep_elitism,                                         # k : 현재 세대의 k개의 best solution만 다음 세대로 이어짐, 0 <= keep_elitism <= sol_per_pop

                    K_tournament = K_tournament,                                         # parent_selection_type이 tournament인 경우에 토너먼트에 참여하는 부모의 수

                    crossover_type = co_type,                                            # 교차 연산 유형, single_point (for single-point crossover), two_points (for two points crossover), uniform (for uniform crossover), and scattered (for scattered crossover)
                    crossover_probability = crossover_probability,   #(논문 : 0.8)        # 교차 연산을 적용할 부모 노드를 선택할 확률, 나머지 확률은 부모 유전자를 그대로 복제해서 다음 세대로 넘김

                    mutation_type = mu_type,                                             # 돌연변이 연산의 유형, random (for random mutation), swap (for swap mutation), inversion (for inversion mutation), scramble (for scramble mutation), and adaptive (for adaptive mutation)
                    mutation_probability = mutation_probability,   #(논문 : 0.02)         # 돌연변이 연산을 적용할 유전자(개체) 선택 확률, 돌연변이 함수 정의 가능, 이 변수가 있으면 mutation_percent_genes와 mutation_num_genes 필요 없음
                    mutation_by_replacement = True,                                      # mutation_type이 random일 때만 작동, True면 기존 유전자를 돌연변이로 대체, False면 기존 유전자에 노이즈 추가
                    mutation_percent_genes = mutation_percent_genes,                     # 돌연변이 대상 개체 내에서 변이할 유전자의 비율 (default : 10%), 여기서 돌연변이할 유전자의 개수가 계산되어 mutation_num_genes에 할당됨
                    mutation_num_genes = None,                                           # 돌연변이할 유전자의 개수 지정, mutation_probability 변수가 있는 경우 작동하지 않음
                    random_mutation_min_val = -0.5,                                      # 유전자에 추가될 난수 값이 선택되는 범위의 하한
                    random_mutation_max_val = 0.5,                                       # 유전자에 추가될 난수 값이 선택되는 범위의 상한

                    gene_space = [{'low': 0, 'high': 1}] * 8,

                    on_start = None,                                                     # 유전 알고리즘이 진화를 시작하기 전에 한 번만 호출되는 함수/메서드
                    on_fitness = None,                                                   # 모집단 내 모든 해의 적합도 값을 계산한 후 호출할 함수/메서드
                    on_parents = None,                                                   # 부모를 선택한 후 호출할 함수/메서드
                    on_crossover = None,                                                 # 교차 연산이 적용될 때마다 호출될 함수
                    on_mutation = None,                                                  # 돌연변이 연산이 적용될 때마다 호출될 함수
                    on_generation = None,                                                # 각 세대마다 호출될 함수
                    on_stop = None,                                                      # 유전 알고리즘이 종료되기 바로 전이나 모든 세대가 완료될 때 한번만 호출되는 함수

                    save_best_solutions = True,                                          # True인 경우 각 세대 이후 best_solution에 최적해 저장
                    save_solutions = True,                                               # 각 세대의 모든 해는 solution에 저장

                    suppress_warnings = False,
                    allow_duplicate_genes = False,                                       # True인 경우, solution/염색체에 중복된 유전자 값이 있을 수 있음

                    stop_criteria = None,
                    parallel_processing = None,                                          # None인 경우 병렬 처리 허용하지 않음

                    random_seed = 42,

                    logger = None                                                        # logger 허용
                    )
    return ga_instance

def run_optimized_ga():
    """최적화된 하이퍼파라미터로 L=100에서 GA를 실행하는 함수"""

    optimized_params = {
        'crossover_type': 'single_point',
        'mutation_type': 'adaptive',
        'parent_selection_type': 'sss',
        'sol_per_pop': 102,
        'num_parents_mating': 22,
        'keep_parents': 21,
        'keep_elitism': 9,
        'crossover_probability': 0.6509333611086074,
        'mutation_percent_genes': [0.5, 0.05]
    }

    print(f"=== L={L}에서 최적화된 하이퍼파라미터로 GA 실행 ===")
    print(f"사용된 하이퍼파라미터:")
    for key, value in optimized_params.items():
        print(f"  {key}: {value}")
    print()
    
    # GA 인스턴스 생성 및 실행
    print("GA를 실행합니다...")
    ga_instance = define_ga(
        co_type=optimized_params['crossover_type'],
        mu_type=optimized_params['mutation_type'],
        sel_type=optimized_params['parent_selection_type'],
        gen = 200,
        num_parents_mating=optimized_params['num_parents_mating'],
        sol_per_pop=optimized_params['sol_per_pop'],
        keep_parents=optimized_params['keep_parents'],
        keep_elitism=optimized_params['keep_elitism'],
        crossover_probability=optimized_params['crossover_probability'],
        mutation_probability=None,  # adaptive mutation 사용
        mutation_percent_genes=optimized_params['mutation_percent_genes'],
        random_seed=42
    )
    
    # GA 실행
    ga_instance.run()
    
    # 결과 추출
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    return solution, solution_fitness, solution_idx

def main():
    """메인 실행 함수"""
    print("최적화된 하이퍼파라미터로 GA를 실행합니다...")
    print("=" * 60)
    
    # 최적화된 GA 실행
    solution, solution_fitness, solution_idx = run_optimized_ga()
    
    print("\n" + "=" * 60)
    print("=== 최종 결과 ===")
    print(f"L = {L}")
    print(f"최적 SKR 값: {solution_fitness:.6e}")
    print()
    
    print("최적 파라미터 값:")
    # 정규화된 솔루션 출력
    normalized_solution = normalize_p(solution)
    print(f"  mu: {normalized_solution[0]:.6f}")
    print(f"  nu: {normalized_solution[1]:.6f}")
    print(f"  vac: {normalized_solution[2]:.6f}")
    print(f"  p_mu: {normalized_solution[3]:.6f}")
    print(f"  p_nu: {normalized_solution[4]:.6f}")
    print(f"  p_vac: {normalized_solution[5]:.6f}")
    print(f"  p_x: {normalized_solution[6]:.6f}")
    print(f"  q_x: {normalized_solution[7]:.6f}")
    

if __name__ == "__main__":
    main()

