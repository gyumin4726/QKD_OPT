"""
QKD 학습 데이터셋 생성기 (파이프라인 1단계)

GA(유전 알고리즘)를 사용하여 다양한 입력 조건에서 최적 QKD 파라미터를 찾고,
이를 학습 데이터셋으로 저장하는 스크립트입니다.

파이프라인:
    1. data_generator.py     → raw_dataset_L{L}.csv 생성
    2. clean_dataset.py      → cleaned_dataset_L{L}.csv 생성
    3. data_split.py         → train_L{L}.csv, test_L{L}.csv 생성
    4. train_fttransformer.py → 모델 학습

주요 기능:
    - 입력 파라미터 조합 생성 (random/grid 샘플링)
    - GA로 각 조합에 대한 최적 파라미터 탐색
    - 학습용 CSV 데이터셋 생성 및 저장

설정:
    - DEFAULT_L: 거리 (km)
    - INCLUDE_Y_0: Y_0를 변수로 사용할지 여부 (False: 0.0 고정)
    - DEFAULT_N_SAMPLES: 생성할 샘플 수
    - SAMPLING_METHOD: 샘플링 방법 ('random' 또는 'grid')

사용법:
    1. 파일 상단의 DEFAULT_L 값 설정 (다른 파일들과 동일하게)
    2. INCLUDE_Y_0 플래그 설정 (train 파일과 동일하게)
    3. python data_generator.py 실행
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from ga_final import define_ga
from simulator import skr_simulator

# ============================================================
# 데이터셋 생성 설정
# ============================================================

# 기본 거리 설정
DEFAULT_L = 100                         # 기본 거리 (km)

# 데이터셋 생성 설정
DEFAULT_N_SAMPLES = 1                   # 생성할 샘플 수
DEFAULT_MAX_GENERATIONS = 100           # GA 최대 세대 수
SAMPLING_METHOD = 'random'              # 샘플링 방법: 'random' 또는 'grid'
RANDOM_SEED = 42                        # 재현성을 위한 랜덤 시드
INCLUDE_Y_0 = False                     # Y_0를 변수로 사용할지 여부 (False: 고정값 0.0 사용)

# 출력 파일 설정
OUTPUT_DIR = 'dataset'                              # 출력 디렉토리
OUTPUT_FILENAME = f'raw_dataset_L{DEFAULT_L}.csv'   # 출력 파일명 (자동 설정)

# ============================================================
# 파라미터 범위 설정
# ============================================================

# 최적화할 파라미터 범위 (입력 변수)
# Y_0 포함 여부는 INCLUDE_Y_0 플래그로 결정
if INCLUDE_Y_0:
    PARAM_RANGES = {
        'eta_d': (0.02, 0.08),              # 탐지기 효율 (2-8%, 기본값 4.5%)
        'Y_0': (0.0, 1e-6),                 # 다크 카운트율 (0 ~ 1e-6)
        'e_d': (0.02, 0.05),                # 오정렬률 (2-5%, 기본값 3.3%)
        'alpha': (0.18, 0.24),              # 광섬유 감쇠 계수 (기본값 0.21)
        'zeta': (1.1, 1.4),                 # 오류 정정 효율 (기본값 1.22)
        'eps_sec': (1e-12, 1e-8),           # 보안 파라미터 (기본값 1e-10)
        'eps_cor': (1e-18, 1e-12),          # 정확성 파라미터 (기본값 1e-15)
        'N': (1e9, 1e11)                    # 광 펄스 수 (기본값 1e10)
    }
    FIXED_PARAMS = {
        'e_0': 0.5                          # 배경 오류율 고정값
    }
else:
    PARAM_RANGES = {
        'eta_d': (0.02, 0.08),              # 탐지기 효율 (2-8%, 기본값 4.5%)
        'e_d': (0.02, 0.05),                # 오정렬률 (2-5%, 기본값 3.3%)
        'alpha': (0.18, 0.24),              # 광섬유 감쇠 계수 (기본값 0.21)
        'zeta': (1.1, 1.4),                 # 오류 정정 효율 (기본값 1.22)
        'eps_sec': (1e-12, 1e-8),           # 보안 파라미터 (기본값 1e-10)
        'eps_cor': (1e-18, 1e-12),          # 정확성 파라미터 (기본값 1e-15)
        'N': (1e9, 1e11)                    # 광 펄스 수 (기본값 1e10)
    }
    FIXED_PARAMS = {
        'e_0': 0.5,                         # 배경 오류율 고정값
        'Y_0': 0.0                          # 다크 카운트율 고정값
    }

# ============================================================
# GA 최적화 설정
# ============================================================

OPTIMIZE_VAC = True                     # vac 파라미터도 최적화할지 여부

# GA 파라미터 (최적화된 설정)
GA_CONFIG = {
    'co_type': 'scattered',             # 교차 방식
    'mu_type': 'adaptive',              # 돌연변이 방식
    'sel_type': 'tournament',           # 선택 방식
    'num_parents_mating': 41,           # 교배할 부모 수
    'sol_per_pop': 188,                 # 인구 크기
    'keep_parents': 40,                 # 유지할 부모 수
    'keep_elitism': 8,                  # 엘리트 유지 수
    'K_tournament': 25,                 # 토너먼트 크기
    'crossover_probability': 0.549428712068568,
    'mutation_probability': None,       # adaptive mutation 사용
    'mutation_percent_genes': [0.7, 0.2],
    'random_seed': 42
}

# ============================================================

class QKDDataGenerator:
    def __init__(self, L=None):
        # L 값 검증 (필수 파라미터)
        if L is None:
            L = DEFAULT_L
            print(f"L 값이 지정되지 않아 기본값 L={L} km 사용")
        
        # L 값 설정
        self.fixed_L = L
        
        # 상단 설정 변수 사용
        self.param_ranges = PARAM_RANGES
        self.fixed_params = FIXED_PARAMS
        
        print(f"L={self.fixed_L} km로 고정하여 데이터셋 생성")
        print(f"Y_0 모드: {'변수 (범위 0 ~ 1e-6)' if INCLUDE_Y_0 else '고정값 (0.0)'}")
    
    def generate_input_combinations(self, n_samples=DEFAULT_N_SAMPLES, method=SAMPLING_METHOD, random_seed=RANDOM_SEED):
        print(f"입력 파라미터 조합 {n_samples}개 생성 중...")
        
        # 시드 설정
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            # 완전히 랜덤하게 하려면 현재 시간 기반으로 시드 설정
            import time
            np.random.seed(int(time.time() * 1000000) % 2**32)
        
        if method == 'random':
            # 무작위 샘플링
            combinations = []
            for _ in tqdm(range(n_samples)):
                combo = {}
                # 고정 파라미터 먼저 추가
                for param, value in self.fixed_params.items():
                    combo[param] = value
                
                # L은 고정값 사용
                combo['L'] = self.fixed_L
                
                # 변수 파라미터 샘플링
                for param, (min_val, max_val) in self.param_ranges.items():
                    if param in ['eps_sec', 'eps_cor', 'N', 'Y_0']:
                        # 로그 스케일로 샘플링 (Y_0 포함)
                        combo[param] = 10 ** np.random.uniform(
                            np.log10(max(min_val, 1e-20)), np.log10(max_val)
                        )
                    else:
                        combo[param] = np.random.uniform(min_val, max_val)
                combinations.append(combo)
        
        elif method == 'grid':
            # 격자 샘플링 (더 체계적)
            # 각 파라미터별로 몇 개의 포인트를 샘플링
            n_per_param = int(n_samples ** (1/len(self.param_ranges)))
            combinations = []
            
            # L은 고정값 사용
            L = self.fixed_L
            
            # e_0는 항상 고정
            e_0 = self.fixed_params['e_0']
            
            # Y_0가 변수인지 고정값인지에 따라 다르게 처리
            if 'Y_0' in self.param_ranges:
                # Y_0가 변수인 경우
                Y_0_range = np.logspace(
                    np.log10(max(self.param_ranges['Y_0'][0], 1e-20)),
                    np.log10(self.param_ranges['Y_0'][1]),
                    n_per_param
                )
            else:
                # Y_0가 고정값인 경우
                Y_0_range = [self.fixed_params['Y_0']]
            
            for eta_d in np.linspace(*self.param_ranges['eta_d'], n_per_param):
                for Y_0 in Y_0_range:
                    for e_d in np.linspace(*self.param_ranges['e_d'], n_per_param):
                        for alpha in np.linspace(*self.param_ranges['alpha'], n_per_param):
                            for zeta in np.linspace(*self.param_ranges['zeta'], n_per_param):
                                for eps_sec in np.logspace(
                                        np.log10(self.param_ranges['eps_sec'][0]),
                                        np.log10(self.param_ranges['eps_sec'][1]),
                                        n_per_param
                                    ):
                                        for eps_cor in np.logspace(
                                            np.log10(self.param_ranges['eps_cor'][0]),
                                            np.log10(self.param_ranges['eps_cor'][1]),
                                            n_per_param
                                        ):
                                            for N in np.logspace(
                                                np.log10(self.param_ranges['N'][0]),
                                                np.log10(self.param_ranges['N'][1]),
                                                n_per_param
                                            ):
                                                combinations.append({
                                                    'L': L, 'eta_d': eta_d, 'Y_0': Y_0,
                                                    'e_d': e_d, 'alpha': alpha, 'zeta': zeta,
                                                    'e_0': e_0, 'eps_sec': eps_sec,
                                                    'eps_cor': eps_cor, 'N': N
                                                })
        
        return combinations[:n_samples]  # 요청한 수만큼만 반환
    
    def optimize_parameters(self, input_params, max_generations=DEFAULT_MAX_GENERATIONS):
        import random
        random.seed(GA_CONFIG['random_seed']) 
        np.random.seed(GA_CONFIG['random_seed'])
        
        # ga_final.py의 전역 OPTIMIZE_VAC 변수 설정
        import ga_final
        ga_final.OPTIMIZE_VAC = OPTIMIZE_VAC
        
        # PyGAD용 래퍼 함수 - 개별 파라미터를 직접 전달
        def skr_fitness_wrapper(ga_instance, solution, solution_idx):
            if OPTIMIZE_VAC:
                # vac도 최적화: 8개 유전자 그대로 사용
                return skr_simulator(ga_instance, solution, solution_idx, **input_params)
            else:
                # vac=0 고정: 7개 유전자를 8개로 확장
                full_solution = np.insert(solution, 2, 0.0)
                return skr_simulator(ga_instance, full_solution, solution_idx, **input_params)
        
        # 상단 GA_CONFIG 사용
        ga = define_ga(
            co_type=GA_CONFIG['co_type'],
            mu_type=GA_CONFIG['mu_type'],
            sel_type=GA_CONFIG['sel_type'],
            gen=max_generations,
            num_parents_mating=GA_CONFIG['num_parents_mating'],
            sol_per_pop=GA_CONFIG['sol_per_pop'],
            keep_parents=GA_CONFIG['keep_parents'],
            keep_elitism=GA_CONFIG['keep_elitism'],
            K_tournament=GA_CONFIG['K_tournament'],
            crossover_probability=GA_CONFIG['crossover_probability'],
            mutation_probability=GA_CONFIG['mutation_probability'],
            mutation_percent_genes=GA_CONFIG['mutation_percent_genes'],
            random_seed=GA_CONFIG['random_seed'],
            fitness_func=skr_fitness_wrapper,
            optimize_vac=OPTIMIZE_VAC  # data_generator의 OPTIMIZE_VAC 사용
        )
        
        # GA 실행
        ga.run()
        solution, solution_fitness, solution_idx = ga.best_solution()
        
        # SKR 계산 (vac 최적화 모드에 따라 처리)
        if OPTIMIZE_VAC:
            # vac도 최적화: 8개 유전자 그대로 사용
            skr_value = skr_simulator(None, solution, 0, **input_params)
            optimal_params = solution.tolist()  # 8개
        else:
            # vac=0 고정: 7개를 8개로 확장
            full_solution = np.insert(solution, 2, 0.0)
            skr_value = skr_simulator(None, full_solution, 0, **input_params)
            optimal_params = full_solution.tolist()  # 8개로 확장
        
        return {
            'optimal_params': optimal_params,
            'skr_value': skr_value,
            'fitness': solution_fitness
        }
    
    def generate_dataset(self, n_samples=DEFAULT_N_SAMPLES, max_generations=DEFAULT_MAX_GENERATIONS, save_path=OUTPUT_FILENAME):
        # 저장 경로 결정
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if save_path is None:
            raise ValueError("출력 파일명을 지정해주세요. 파일 상단의 OUTPUT_FILENAME을 설정하세요.")
        
        # 파일명이 지정되면 OUTPUT_DIR 안에 저장
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        
        print(f"QKD 데이터셋 생성 시작 (L={self.fixed_L} km 고정, 샘플 수: {n_samples})")
        
        # 입력 조합 생성 (재현 가능성을 위해 시드 고정)
        input_combinations = self.generate_input_combinations(n_samples, method=SAMPLING_METHOD, random_seed=RANDOM_SEED)
        
        # 각 조합에 대해 최적화 수행
        dataset = []
        failed_count = 0
        
        for i, input_params in enumerate(tqdm(input_combinations, desc="최적화 진행")):
            try:
                # 최적화 수행
                result = self.optimize_parameters(input_params, max_generations)
                
                # 데이터 포인트 생성
                # INCLUDE_Y_0 플래그에 따라 Y_0 포함 여부 결정
                data_point = {
                    # 입력 파라미터들 (L, e_0 제외, Y_0는 플래그에 따라)
                    'eta_d': input_params['eta_d'],
                }
                
                # Y_0가 변수인 경우 데이터셋에 포함
                if INCLUDE_Y_0:
                    data_point['Y_0'] = input_params['Y_0']
                
                # 나머지 입력 파라미터
                data_point.update({
                    'e_d': input_params['e_d'],
                    'alpha': input_params['alpha'],
                    'zeta': input_params['zeta'],
                    'eps_sec': input_params['eps_sec'],
                    'eps_cor': input_params['eps_cor'],
                    'N': input_params['N'],
                    
                    # 출력 파라미터들 (8개 QKD 파라미터)
                    'mu': result['optimal_params'][0],
                    'nu': result['optimal_params'][1],
                    'vac': result['optimal_params'][2],
                    'p_mu': result['optimal_params'][3],
                    'p_nu': result['optimal_params'][4],
                    'p_vac': result['optimal_params'][5],
                    'p_X': result['optimal_params'][6],
                    'q_X': result['optimal_params'][7],
                    
                    # SKR (fitness 제거)
                    'skr': result['skr_value']
                })
                dataset.append(data_point)
                
                # 진행 상황 출력
                if (i + 1) % 100 == 0:
                    print(f"완료: {i+1}/{n_samples}, 실패: {failed_count}")
                    
            except Exception as e:
                print(f"최적화 실패 (인덱스 {i}): {e}")
                failed_count += 1
                continue
        
        print(f"데이터셋 생성 완료!")
        print(f"성공: {len(dataset)}, 실패: {failed_count}")
        
        # DataFrame으로 변환하고 CSV 저장
        df = pd.DataFrame(dataset)
        df.to_csv(save_path, index=False)
        print(f"데이터셋이 {save_path}에 저장되었습니다.")
        
        return dataset
    
    def load_dataset(self, path='qkd_dataset.csv'):
        if path.endswith('.csv'):
            return pd.read_csv(path)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def analyze_dataset(self, dataset):
        print("=== 데이터셋 분석 ===")
        
        # DataFrame인지 리스트인지 확인
        if isinstance(dataset, pd.DataFrame):
            df = dataset
            print(f"총 샘플 수: {len(df)}")
            
            if len(df) == 0:
                print("경고: 데이터셋이 비어있습니다.")
                return
            
            # SKR 분포 분석
            skr_values = df['skr'].values
            print(f"SKR 범위: {min(skr_values):.2e} ~ {max(skr_values):.2e}")
            print(f"SKR 평균: {np.mean(skr_values):.2e}")
            print(f"SKR 중앙값: {np.median(skr_values):.2e}")
                
        else:
            # 기존 리스트 형태
            print(f"총 샘플 수: {len(dataset)}")
            
            if len(dataset) == 0:
                print("경고: 데이터셋이 비어있습니다.")
                return
            
            # SKR 분포 분석
            skr_values = [d['skr'] for d in dataset]
            print(f"SKR 범위: {min(skr_values):.2e} ~ {max(skr_values):.2e}")
            print(f"SKR 평균: {np.mean(skr_values):.2e}")
            print(f"SKR 중앙값: {np.median(skr_values):.2e}")

if __name__ == "__main__":
    # 상단 설정으로 데이터 생성기 초기화
    generator = QKDDataGenerator(L=DEFAULT_L)
    
    # 데이터셋 생성 (상단 설정 사용)
    print("데이터셋 생성 중...")
    test_dataset = generator.generate_dataset(
        n_samples=DEFAULT_N_SAMPLES,
        max_generations=DEFAULT_MAX_GENERATIONS,
        save_path=OUTPUT_FILENAME 
    )
    
    # 데이터셋 분석
    generator.analyze_dataset(test_dataset)