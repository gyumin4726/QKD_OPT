"""
QKD 파라미터 최적화 - 최적화된 GA 실행

최적화된 GA 하이퍼파라미터를 사용하여 QKD 파라미터를 탐색하는 스크립트입니다.
ga_final.py에서 찾은 최적 하이퍼파라미터로 단일 실행합니다.

주요 기능:
    - 사전 최적화된 GA 하이퍼파라미터 사용
    - QKD 파라미터 탐색 및 결과 출력
    - vac 파라미터 최적화 모드 지원

사용법:
    1. 파일 상단의 설정 변수 수정 (L, 환경 변수, 하이퍼파라미터 등)
    2. python ga_crosscheck.py 실행
"""

import numpy as np
import pygad

import warnings
warnings.filterwarnings('ignore')

from simulator import skr_simulator

# ============================================================
# 최적화 설정
# ============================================================

L = 20                              # 광섬유 길이 (km)
OPTIMIZE_VAC = False                # True: vac도 최적화, False: vac=0 고정

# 최적화된 GA 하이퍼파라미터
OPTIMIZED_PARAMS = {
    'crossover_type': 'scattered',
    'mutation_type': 'adaptive',
    'parent_selection_type': 'tournament',
    'sol_per_pop': 188,
    'num_parents_mating': 41,
    'keep_parents': 40,
    'keep_elitism': 8,
    'crossover_probability': 0.549428712068568,
    'mutation_probability': None,
    'mutation_percent_genes': [0.7, 0.2],
    'K_tournament': 25
}

# ============================================================
# 환경 변수 설정
# ============================================================

ETA_D = 0.045                       # 단일 광자 검출기의 검출 효율 (%)
Y_0 = 1.7e-6                        # 암계수율 (dark count rate)
E_D = 0.033                         # 정렬 오류율 (misalignment rate)
ALPHA = 0.21                        # 단일 모드 광섬유의 감쇠 계수
ZETA = 1.22                         # 오류 정정 효율
E_0 = 0.5                           # 배경 오류율
EPS_SEC = 1.0e-10                   # 보안 매개변수
EPS_COR = 1.0e-15                   # 정확성 매개변수
N = 1.0e10                          # Alice가 보낸 광 펄스 개수
LAMBDA = None                       # Xk에서 관찰된 비트 값 1의 확률

# ============================================================

print(f"vac 최적화 모드: {'ON (vac 포함)' if OPTIMIZE_VAC else 'OFF (vac=0 고정)'}")

# 재현 가능한 결과를 위한 시드 설정
import random
random.seed(42)
np.random.seed(42)

def skr_fitness_wrapper(ga_instance, solution, solution_idx):
    if OPTIMIZE_VAC:
        return skr_simulator(
            ga_instance, solution, solution_idx,
            L=L, eta_d=ETA_D, Y_0=Y_0, e_d=E_D,
            alpha=ALPHA, zeta=ZETA, e_0=E_0,
            eps_sec=EPS_SEC, eps_cor=EPS_COR, N=N, Lambda=LAMBDA
        )
    else:
        full_solution = np.insert(solution, 2, 0.0)
        return skr_simulator(
            ga_instance, full_solution, solution_idx,
            L=L, eta_d=ETA_D, Y_0=Y_0, e_d=E_D,
            alpha=ALPHA, zeta=ZETA, e_0=E_0,
            eps_sec=EPS_SEC, eps_cor=EPS_COR, N=N, Lambda=LAMBDA
        )

def define_ga(co_type, mu_type, sel_type, 
              gen = 200,
              num_parents_mating = 60, sol_per_pop = 200, keep_parents = 50, keep_elitism = 10, K_tournament = 8, crossover_probability = 0.8, mutation_probability = 0.02, mutation_percent_genes = "default",
              random_seed = 42):
    
    num_genes = 8 if OPTIMIZE_VAC else 7
    
    ga_instance = pygad.GA(
                    num_generations = gen,
                    num_parents_mating = num_parents_mating,
                    fitness_func = skr_fitness_wrapper,
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
                    random_seed = 42,
                    logger = None
                    )
    return ga_instance

def run_optimized_ga():
    print(f"=== L={L}에서 최적화된 하이퍼파라미터로 GA 실행 ===")
    print(f"사용된 하이퍼파라미터:")
    for key, value in OPTIMIZED_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    
    print("GA를 실행합니다...")
    
    if OPTIMIZED_PARAMS['mutation_type'] == 'adaptive':
        mutation_percent = OPTIMIZED_PARAMS['mutation_percent_genes']
        mutation_prob = None
    else:
        mutation_prob = OPTIMIZED_PARAMS['mutation_probability'] if OPTIMIZED_PARAMS['mutation_probability'] is not None else 0.02
        mutation_percent = 'default'
    
    ga_instance = define_ga(
        co_type=OPTIMIZED_PARAMS['crossover_type'],
        mu_type=OPTIMIZED_PARAMS['mutation_type'],
        sel_type=OPTIMIZED_PARAMS['parent_selection_type'],
        gen = 200,
        num_parents_mating=OPTIMIZED_PARAMS['num_parents_mating'],
        sol_per_pop=OPTIMIZED_PARAMS['sol_per_pop'],
        keep_parents=OPTIMIZED_PARAMS['keep_parents'],
        keep_elitism=OPTIMIZED_PARAMS['keep_elitism'],
        crossover_probability=OPTIMIZED_PARAMS['crossover_probability'],
        mutation_probability=mutation_prob,
        mutation_percent_genes=mutation_percent,
        K_tournament=OPTIMIZED_PARAMS['K_tournament'],
        random_seed=42
    )
    
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    return solution, solution_fitness, solution_idx

def main():
    print("최적화된 하이퍼파라미터로 GA를 실행합니다...")
    print("=" * 60)
    
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