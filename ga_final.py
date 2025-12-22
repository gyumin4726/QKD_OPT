"""
QKD 파라미터 최적화 - GA 하이퍼파라미터 튜닝

Optuna를 사용하여 GA(유전 알고리즘) 하이퍼파라미터를 최적화하고,
최적화된 설정으로 QKD 파라미터를 탐색하는 스크립트입니다.

주요 기능:
    - Optuna를 이용한 GA 하이퍼파라미터 최적화
    - 최적화된 GA로 QKD 파라미터 탐색
    - vac 파라미터 최적화 모드 지원

사용법:
    1. 파일 상단의 설정 변수 수정 (L, 환경 변수 등)
    2. python ga_final.py 실행
"""

import numpy as np
import os
import pygad
import time
from tqdm import tqdm
import optuna

import warnings
warnings.filterwarnings('ignore')

# SKR 시뮬레이터 import
from simulator import skr_simulator

# ============================================================
# 최적화 설정
# ============================================================

# 거리 설정
L = 20                              # 광섬유 길이 (km)

# vac 최적화 모드 설정
OPTIMIZE_VAC = False                # True: vac도 최적화, False: vac=0 고정

# Optuna 최적화 설정
N_TRIALS = 1000                     # Optuna 최적화 시도 횟수
NUM_ITER = 1                        # 각 시도당 GA 반복 횟수

# 초기 하이퍼파라미터 (Optuna 시작점)
INITIAL_PARAMS = {
    'crossover_type': 'two_points',
    'mutation_type': 'adaptive',
    'parent_selection_type': 'tournament',
    'sol_per_pop': 239,
    'num_parents_mating': 125,
    'keep_parents': 107,
    'keep_elitism': 12,
    'crossover_probability': 0.7248539327369946,
    'mutation_percent_genes': [0.3, 0.1],
    'K_tournament': 71
}

# ============================================================
# 환경 변수 설정 (ga_final.py 단독 실행 시에만 사용)
# ============================================================
# 주의: data_generator.py에서 import하여 사용할 때는
#       data_generator.py의 설정이 우선됩니다.
#       아래 환경 변수는 ga_final.py를 직접 실행할 때만 사용됩니다.

# 검출 파라미터
ETA_D = 0.045                       # 단일 광자 검출기의 검출 효율 (%)
Y_0 = 1.7e-6                        # 암계수율 (dark count rate)
E_D = 0.033                         # 정렬 오류율 (misalignment rate)

# 광섬유 파라미터
ALPHA = 0.21                        # 단일 모드 광섬유의 감쇠 계수

# 오류 정정 파라미터
ZETA = 1.22                         # 오류 정정 효율
E_0 = 0.5                           # 배경 오류율

# 보안 파라미터
EPS_SEC = 1.0e-10                   # 보안 매개변수
EPS_COR = 1.0e-15                   # 정확성 매개변수

# 시스템 파라미터
N = 1.0e10                          # Alice가 보낸 광 펄스 개수
LAMBDA = None                       # Xk에서 관찰된 비트 값 1의 확률

# ============================================================

# CPU 코어 수 자동 감지
CPU_COUNT = os.cpu_count()

# 재현 가능한 결과를 위한 시드 설정
import random
random.seed(42)
np.random.seed(42)

def define_ga(co_type, mu_type, sel_type, 
              gen = 200,
              num_parents_mating = 60, sol_per_pop = 200, keep_parents = 50, keep_elitism = 10, K_tournament = 8, crossover_probability = 0.8, mutation_probability = 0.02, mutation_percent_genes = "default",
              random_seed = 42, fitness_func=None, optimize_vac=None):
    
    # optimize_vac이 지정되지 않으면 전역 변수 사용
    if optimize_vac is None:
        optimize_vac = OPTIMIZE_VAC
    
    # 유전자 개수 설정
    num_genes = 8 if optimize_vac else 7
    
    # fitness_func이 제공되지 않은 경우 기본 래퍼 함수 생성
    if fitness_func is None:
        if optimize_vac:
            def skr_fitness_wrapper(ga_instance, solution, solution_idx):
                return skr_simulator(
                    ga_instance, solution, solution_idx,
                    L=L, eta_d=ETA_D, Y_0=Y_0, e_d=E_D,
                    alpha=ALPHA, zeta=ZETA, e_0=E_0,
                    eps_sec=EPS_SEC, eps_cor=EPS_COR, N=N, Lambda=LAMBDA
                )
        else:
            def skr_fitness_wrapper(ga_instance, solution, solution_idx):
                full_solution = np.insert(solution, 2, 0.0)
                return skr_simulator(
                    ga_instance, full_solution, solution_idx,
                    L=L, eta_d=ETA_D, Y_0=Y_0, e_d=E_D,
                    alpha=ALPHA, zeta=ZETA, e_0=E_0,
                    eps_sec=EPS_SEC, eps_cor=EPS_COR, N=N, Lambda=LAMBDA
                )
        fitness_func = skr_fitness_wrapper
    
    ga_instance = pygad.GA(
                    num_generations = gen,
                    num_parents_mating = num_parents_mating,
                    fitness_func = fitness_func,
                    fitness_batch_size = None,
                    initial_population = None,
                    sol_per_pop = sol_per_pop,
                    num_genes = num_genes,
                    gene_type = [float, 6],
                    init_range_low = 0,
                    init_range_high = 1,
                    parent_selection_type = sel_type,
                    keep_parents = keep_parents,
                    keep_elitism = keep_elitism,
                    K_tournament = K_tournament,
                    crossover_type = co_type,
                    crossover_probability = crossover_probability,
                    mutation_type = mu_type,
                    mutation_probability = mutation_probability,
                    mutation_by_replacement = True,
                    mutation_percent_genes = mutation_percent_genes,
                    mutation_num_genes = None,
                    random_mutation_min_val = -0.5,
                    random_mutation_max_val = 0.5,
                    gene_space = [{'low': 0, 'high': 1}] * num_genes,
                    on_start = None,
                    on_fitness = None,
                    on_parents = None,
                    on_crossover = None,
                    on_mutation = None,
                    on_generation = None,
                    on_stop = None,
                    save_best_solutions = True,
                    save_solutions = True,
                    suppress_warnings = False,
                    allow_duplicate_genes = False,
                    stop_criteria = None,
                    parallel_processing = None,
                    random_seed = random_seed,
                    logger = None
                    )
    return ga_instance

def objective(trial):
    total_fitness = 0
    
    crossover_type = trial.suggest_categorical("crossover_type", ["single_point", "two_points", "uniform", "scattered"])
    mutation_type = trial.suggest_categorical("mutation_type", ["random", "swap", "inversion", "scramble", "adaptive"])
    parent_selection_type = trial.suggest_categorical("parent_selection_type", ["sss", "rws", "sus", "rank", "random", "tournament"])

    sol_per_pop = trial.suggest_int("sol_per_pop", 80, 250)
    num_parents_mating = trial.suggest_int("num_parents_mating", int(sol_per_pop*0.2), sol_per_pop)
    keep_parents = trial.suggest_int("keep_parents", 1, num_parents_mating)
    keep_elitism = trial.suggest_int("keep_elitism", 0, 20)    
    crossover_probability = trial.suggest_float("crossover_probability", 0.2, 1)

    if mutation_type == "adaptive":
        mutation_percent_genes = trial.suggest_categorical("mutation_percent_genes", [[0.5, 0.05], [0.3, 0.1], [0.7, 0.2]])
        mutation_probability = None
    else:
        mutation_percent_genes = "default"
        mutation_probability = trial.suggest_float("mutation_probability", 0.01, 0.5)

    K_tournament = trial.suggest_int("K_tournament", 2, int(num_parents_mating * 0.7)) if parent_selection_type == "tournament" else None
    for _ in range(NUM_ITER):
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
                       random_seed=42,
                       fitness_func=None)

        ga.run()
        best_fitness = ga.best_solution()[1]
        total_fitness += best_fitness

    return - total_fitness 

def run_optimization():
    sampler = optuna.samplers.TPESampler(n_startup_trials=20,  
                                         multivariate=False,    
                                         group=False)

    study = optuna.create_study(sampler = sampler, direction="minimize")
    
    study.enqueue_trial(INITIAL_PARAMS)
    print("초기 하이퍼파라미터를 시작점으로 추가했습니다.")
    
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    print("Best trial:")
    print(study.best_trial)
    
    return study

def run_final_ga(study):
    print(f"L={L}에서 최적화된 하이퍼파라미터로 GA를 실행합니다...")
    
    mutation_type = study.best_trial.params['mutation_type']
    if mutation_type == "adaptive":
        mutation_percent_genes = study.best_trial.params['mutation_percent_genes']
        mutation_probability = None
    else:
        mutation_percent_genes = "default"
        mutation_probability = study.best_trial.params['mutation_probability']
    
    parent_selection_type = study.best_trial.params['parent_selection_type']
    if parent_selection_type == "tournament":
        K_tournament = study.best_trial.params['K_tournament']
    else:
        K_tournament = None
    
    for i in tqdm(range(NUM_ITER)) : 
        ga_instance = define_ga(co_type = study.best_trial.params['crossover_type'], 
                                mu_type = mutation_type, 
                                sel_type = parent_selection_type, 
                                gen = 200,
                                num_parents_mating = study.best_trial.params['num_parents_mating'], 
                                sol_per_pop = study.best_trial.params['sol_per_pop'],
                                keep_parents = study.best_trial.params['keep_parents'], 
                                keep_elitism = study.best_trial.params['keep_elitism'], 
                                crossover_probability = study.best_trial.params['crossover_probability'], 
                                mutation_probability = mutation_probability,
                                mutation_percent_genes = mutation_percent_genes, 
                                K_tournament = K_tournament,
                                random_seed = 42,
                                fitness_func = None)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution_fitness, solution, solution_fitness, solution_idx

def main():
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