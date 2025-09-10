import numpy as np
import os
import pygad
import time
from tqdm import tqdm
import optuna
import yaml

import warnings
warnings.filterwarnings('ignore')

# SKR 시뮬레이터 import
from simulator import skr_simulator

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
Lambda = config['system']['Lambda']

# 광섬유 길이 L 사용자 입력
L = 100

import simulator
simulator.L = L

# 파생 상수
eps = eps_sec/23
beta = np.log(1/eps)

# CPU 코어 수 자동 감지
CPU_COUNT = os.cpu_count()
print(f"사용 가능한 CPU 코어 수: {CPU_COUNT}")

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

                    fitness_func = skr_simulator,
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

                    random_seed = random_seed,

                    logger = None                                                        # logger 허용
                    )
    return ga_instance

num_iter = 1

def objective(trial):
    """Optuna 최적화 목적 함수"""
    total_fitness = 0
    
    crossover_type = trial.suggest_categorical("crossover_type", ["single_point", "two_points", "uniform", "scattered"])
    mutation_type = trial.suggest_categorical("mutation_type", ["random", "swap", "inversion", "scramble", "adaptive"])
    parent_selection_type = trial.suggest_categorical("parent_selection_type", ["sss", "rws", "sus", "rank", "random", "tournament"])

    sol_per_pop = trial.suggest_int("sol_per_pop", 80, 250)
    num_parents_mating = trial.suggest_int("num_parents_mating", int(sol_per_pop*0.2), sol_per_pop)
    keep_parents = trial.suggest_int("keep_parents", 1, num_parents_mating)
    keep_elitism = trial.suggest_int("keep_elitism", 0, 20)    
    crossover_probability = trial.suggest_float("crossover_probability", 0.2, 1)

    # mutation 
    if mutation_type == "adaptive":
        mutation_percent_genes = trial.suggest_categorical("mutation_percent_genes", [[0.5, 0.05], [0.3, 0.1], [0.7, 0.2]])
        mutation_probability = None
    else:
        mutation_percent_genes = "default"
        mutation_probability = trial.suggest_float("mutation_probability", 0.01, 0.5)

    # tournament
    K_tournament = trial.suggest_int("K_tournament", 2, int(num_parents_mating * 0.7)) if parent_selection_type == "tournament" else None

    # 고정된 L에 대하여 최적화
    for _ in range(num_iter):
        ga = define_ga(co_type=crossover_type,
                       mu_type=mutation_type,
                       sel_type=parent_selection_type,
                       gen = 200,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       K_tournament=K_tournament,
                       crossover_probability=crossover_probability,
                       mutation_probability=mutation_probability,
                       mutation_percent_genes=mutation_percent_genes,
                       random_seed=42)

        ga.run()
        best_fitness = ga.best_solution()[1]
        total_fitness += best_fitness

    return - total_fitness 

def run_optimization():
    """Optuna 최적화를 실행하는 함수 - 최적화된 하이퍼파라미터를 초기값으로 사용"""
    sampler = optuna.samplers.TPESampler(n_startup_trials=20,  
                                         multivariate=False,    
                                         group=False)

    study = optuna.create_study(sampler = sampler, direction="minimize")
    
    # 최적화된 하이퍼파라미터를 초기 시도로 추가
    initial_params = {
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
    
    # 초기 시도를 study에 추가
    study.enqueue_trial(initial_params)
    print("최적화된 하이퍼파라미터를 초기 시도로 추가했습니다.")
    
    study.optimize(objective, n_trials=1, n_jobs=1)

    print("Best trial:")
    print(study.best_trial)
    
    return study

def run_final_ga(study):
    """최적화된 하이퍼파라미터로 최종 GA를 실행하는 함수"""
    num_iter = 1

    print(f"L={L}에서 최적화된 하이퍼파라미터로 GA를 실행합니다...")
    
    for i in tqdm(range(num_iter)) : 
        ga_instance = define_ga(co_type = study.best_trial.params['crossover_type'], 
                                mu_type = study.best_trial.params['mutation_type'], 
                                sel_type = study.best_trial.params['parent_selection_type'], 
                                gen = 200,
                                num_parents_mating = study.best_trial.params['num_parents_mating'], 
                                sol_per_pop = study.best_trial.params['sol_per_pop'],
                                keep_parents = study.best_trial.params['keep_parents'], 
                                keep_elitism = study.best_trial.params['keep_elitism'], 
                                crossover_probability = study.best_trial.params['crossover_probability'], 
                                mutation_probability = None,
                                mutation_percent_genes = study.best_trial.params['mutation_percent_genes'], 
                                random_seed = 42)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution_fitness, solution, solution_fitness, solution_idx

def main():
    """메인 실행 함수 - L=100 특화, 최적화된 하이퍼파라미터를 초기값으로 사용"""
    start_time = time.time()
    
    print("=" * 60)
    print("QKD 파라미터 최적화 시작")
    print("=" * 60)
    print(f"CPU 코어 수: {CPU_COUNT}")
    print(f"광섬유 길이 L: {L}")
    print("=" * 60)
    
    print(f"\nL={L}에서 Optuna를 사용한 하이퍼파라미터 최적화를 시작합니다...")
    opt_start = time.time()
    study = run_optimization()
    opt_time = time.time() - opt_start
    print(f"Optuna 최적화 완료: {opt_time:.2f}초")
    
    print(f"\n최적화된 하이퍼파라미터로 L={L}에서 최종 GA를 실행합니다...")
    ga_start = time.time()
    skr_value, solution, solution_fitness, solution_idx = run_final_ga(study)
    ga_time = time.time() - ga_start
    print(f"GA 실행 완료: {ga_time:.2f}초")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"L={L}에서의 최적화 결과")
    print(f"{'='*60}")
    print(f"최적 SKR 값: {skr_value:.6e}")
    print(f"최적 솔루션: {solution}")
    print(f"총 실행 시간: {total_time:.2f}초")
    print(f" - Optuna 최적화: {opt_time:.2f}초 ({opt_time/total_time*100:.1f}%)")
    print(f" - GA 실행: {ga_time:.2f}초 ({ga_time/total_time*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

