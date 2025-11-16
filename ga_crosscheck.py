import numpy as np
import pygad
import yaml

import warnings
warnings.filterwarnings('ignore')

# SKR 시뮬레이터 import
from simulator import skr_simulator, normalize_p, QKDSimulatorConfig

# L은 직접 설정
L = 20

# vac 최적화 모드 설정 (True: vac도 최적화, False: vac=0 고정)
OPTIMIZE_VAC = False

# 설정 객체 생성 (YAML에서 자동 로드)
simulator_config = QKDSimulatorConfig.from_yaml('config/config.yaml')
simulator_config.L = L  # L 값만 별도로 설정

print(f"vac 최적화 모드: {'ON (vac 포함)' if OPTIMIZE_VAC else 'OFF (vac=0 고정)'}")

# 재현 가능한 결과를 위한 시드 설정
import random
random.seed(42)
np.random.seed(42)

# PyGAD용 래퍼 함수 (고정된 시그니처 필요)
def skr_fitness_wrapper(ga_instance, solution, solution_idx):
    """PyGAD용 SKR 적합도 함수 래퍼 - vac 최적화 모드 지원"""
    if OPTIMIZE_VAC:
        # vac도 최적화: 8개 유전자 그대로 사용
        return skr_simulator(ga_instance, solution, solution_idx, simulator_config)
    else:
        # vac=0 고정: 7개 유전자를 8개로 확장
        # solution은 7개: mu, nu, p_mu, p_nu, p_vac, p_X, q_X
        # vac=0을 인덱스 2에 삽입하여 8개로 만듦
        full_solution = np.insert(solution, 2, 0.0)
        return skr_simulator(ga_instance, full_solution, solution_idx, simulator_config)

def define_ga(co_type, mu_type, sel_type, 
              gen = 200,
              num_parents_mating = 60, sol_per_pop = 200, keep_parents = 50, keep_elitism = 10, K_tournament = 8, crossover_probability = 0.8, mutation_probability = 0.02, mutation_percent_genes = "default",
              random_seed = 42):
    """유전 알고리즘 인스턴스를 정의하는 함수 - vac 최적화 모드 지원"""
    
    # 유전자 개수 설정 (OPTIMIZE_VAC에 따라)
    num_genes = 8 if OPTIMIZE_VAC else 7
    
    ga_instance = pygad.GA(num_generations = gen,   #(논문 : 최대 1000)                    # 세대 수
                    num_parents_mating = num_parents_mating,   #(논문 : 30)               # 부모로 선택될 솔루션의 수

                    fitness_func = skr_fitness_wrapper,
                    fitness_batch_size = None,                                           # 배치 단위로 적합도 함수를 계산, 적합도 함수는 각 배치에 대해 한 번씩 호출

                    initial_population = None,                                           # 사용자 정의 초기 개체군, num_genes와 크기가 같아야 함
                    sol_per_pop = sol_per_pop,                                           # 한 세대에 포함되는 솔루션(염색체)의 수, 크면 탐색 다양성이 높아짐, 작으면 빠르게 수렴하지만 최적해를 놓칠 수 있음, initial population이 있으면 작동하지 않음
                    num_genes = num_genes,                                               # 염색체 내 유전자 수 (OPTIMIZE_VAC에 따라 7 또는 8), initial_population을 사용하는 경우 이 매개변수가 필요하지 않음
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

                    gene_space = [{'low': 0, 'high': 1}] * num_genes,                    # 유전자 공간 (OPTIMIZE_VAC에 따라 7 또는 8개)

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
        'crossover_type': 'scattered',
        'mutation_type': 'adaptive',
        'parent_selection_type': 'tournament',
        'sol_per_pop': 188,
        'num_parents_mating': 41,
        'keep_parents': 40,
        'keep_elitism': 8,
        'crossover_probability': 0.549428712068568,
        'mutation_probability': None,  # adaptive mutation은 probability 대신 percent_genes 사용
        'mutation_percent_genes': [0.7, 0.2],
        'K_tournament': 25
    }

    print(f"=== L={L}에서 최적화된 하이퍼파라미터로 GA 실행 ===")
    print(f"사용된 하이퍼파라미터:")
    for key, value in optimized_params.items():
        print(f"  {key}: {value}")
    print()
    
    # GA 인스턴스 생성 및 실행
    print("GA를 실행합니다...")
    
    # adaptive mutation의 경우 mutation_percent_genes에 리스트를 전달
    if optimized_params['mutation_type'] == 'adaptive':
        mutation_percent = optimized_params['mutation_percent_genes']
        mutation_prob = None
    else:
        mutation_prob = optimized_params['mutation_probability'] if optimized_params['mutation_probability'] is not None else 0.02
        mutation_percent = 'default'
    
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
        mutation_probability=mutation_prob,
        mutation_percent_genes=mutation_percent,
        K_tournament=optimized_params['K_tournament'],
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
    
    if OPTIMIZE_VAC:
        print("최적 파라미터 값 (vac 최적화):")
        print(f"  mu: {solution[0]:.6f}")
        print(f"  nu: {solution[1]:.6f}")
        print(f"  vac: {solution[2]:.6f}")
        print(f"  p_mu: {solution[3]:.6f}")
        print(f"  p_nu: {solution[4]:.6f}")
        print(f"  p_vac: {solution[5]:.6f}")
        print(f"  p_x: {solution[6]:.6f}")
        print(f"  q_x: {solution[7]:.6f}")
    else:
        print("최적 파라미터 값 (vac=0 고정):")
        print(f"  mu: {solution[0]:.6f}")
        print(f"  nu: {solution[1]:.6f}")
        print(f"  vac: 0.000000 (고정)")
        print(f"  p_mu: {solution[2]:.6f}")
        print(f"  p_nu: {solution[3]:.6f}")
        print(f"  p_vac: {solution[4]:.6f}")
        print(f"  p_x: {solution[5]:.6f}")
        print(f"  q_x: {solution[6]:.6f}")
    

if __name__ == "__main__":
    main()

