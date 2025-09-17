import numpy as np
import yaml
import pandas as pd
from tqdm import tqdm
import os
import pickle
from ga_final import define_ga
from simulator import skr_simulator

class QKDDataGenerator:
    def __init__(self, config_path='config/config_crosscheck.yaml'):
        """QKD 학습 데이터 생성기"""
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 기본 파라미터 범위 설정 (config.yaml 기준)
        self.param_ranges = {
            'L': (0, 150),                    # 광섬유 길이 (km) - 0~150까지 10단위
            'eta_d': (0.02, 0.08),           # 탐지기 효율 (2-8%, 기본값 4.5%)
            'Y_0': (1e-7, 1e-5),             # 다크 카운트율 (기본값 1.7e-6)
            'e_d': (0.02, 0.05),             # 오정렬률 (2-5%, 기본값 3.3%)
            'alpha': (0.18, 0.24),           # 광섬유 감쇠 계수 (기본값 0.21)
            'zeta': (1.1, 1.4),              # 오류 정정 효율 (기본값 1.22)
            'e_0': (0.4, 0.6),               # 배경 오류율 (기본값 0.5)
            'eps_sec': (1e-12, 1e-8),        # 보안 파라미터 (기본값 1e-10)
            'eps_cor': (1e-18, 1e-12),       # 정확성 파라미터 (기본값 1e-15)
            'N': (1e9, 1e11)                 # 광 펄스 수 (기본값 1e10)
        }
        
        # L 값 리스트 (0, 10, 20, ..., 150)
        self.L_values = list(range(0, 151, 10))
        
        # 고정 파라미터 (현재는 없음, 모든 파라미터를 변수로 사용)
        self.fixed_params = {}
    
    def generate_input_combinations(self, n_samples=10000, method='random', random_seed=None):
        """다양한 입력 파라미터 조합 생성"""
        print(f"입력 파라미터 조합 {n_samples}개 생성 중...")
        
        # 시드 설정 (None이면 랜덤, 숫자면 고정)
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
                for param, (min_val, max_val) in self.param_ranges.items():
                    if param in self.fixed_params:
                        combo[param] = self.fixed_params[param]
                    elif param == 'L':
                        # L은 10단위로 고정 (0, 10, 20, ..., 150)
                        combo[param] = np.random.choice(self.L_values)
                    else:
                        if param in ['eps_sec', 'eps_cor', 'N', 'Y_0']:
                            # 로그 스케일로 샘플링
                            combo[param] = 10 ** np.random.uniform(
                                np.log10(min_val), np.log10(max_val)
                            )
                        else:
                            combo[param] = np.random.uniform(min_val, max_val)
                combinations.append(combo)
        
        elif method == 'grid':
            # 격자 샘플링 (더 체계적)
            # 각 파라미터별로 몇 개의 포인트를 샘플링
            n_per_param = int(n_samples ** (1/len(self.param_ranges)))
            combinations = []
            
            for L in np.linspace(*self.param_ranges['L'], n_per_param):
                for eta_d in [self.fixed_params['eta_d']]:
                    for Y_0 in np.logspace(
                        np.log10(self.param_ranges['Y_0'][0]),
                        np.log10(self.param_ranges['Y_0'][1]),
                        n_per_param
                    ):
                        for e_d in np.linspace(*self.param_ranges['e_d'], n_per_param):
                            for alpha in np.linspace(*self.param_ranges['alpha'], n_per_param):
                                for zeta in np.linspace(*self.param_ranges['zeta'], n_per_param):
                                    for e_0 in np.linspace(*self.param_ranges['e_0'], n_per_param):
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
    
    def optimize_parameters(self, input_params, max_generations=100):
        """주어진 입력 파라미터에 대해 GA로 최적 파라미터 찾기"""
        import random
        random.seed(42) 
        np.random.seed(42)
        
        # PyGAD용 래퍼 함수 - 개별 파라미터를 직접 전달
        def skr_fitness_wrapper(ga_instance, solution, solution_idx):
            return skr_simulator(ga_instance, solution, solution_idx, **input_params)
        
        # 최적화된 GA 설정 (README.md에서 가져온 설정)
        ga = define_ga(
            co_type='single_point',
            mu_type='adaptive', 
            sel_type='sss',
            gen=max_generations,
            num_parents_mating=22,
            sol_per_pop=102,
            keep_parents=21,
            keep_elitism=9,
            K_tournament=None,  # sss 선택에서는 사용하지 않음
            crossover_probability=0.6509333611086074,
            mutation_probability=None,  # adaptive mutation 사용
            mutation_percent_genes=[0.5, 0.05],
            random_seed=42,
            fitness_func=skr_fitness_wrapper  # 래퍼 함수 사용
        )
        
        # GA 실행
        ga.run()
        solution, solution_fitness, solution_idx = ga.best_solution()
        
        # SKR 계산
        skr_value = skr_simulator(None, solution, 0, **input_params)
        
        return {
            'optimal_params': solution.tolist(),
            'skr_value': skr_value,
            'fitness': solution_fitness
        }
    
    def generate_dataset(self, n_samples=1000, max_generations=50, save_path='qkd_dataset.csv'):
        """전체 데이터셋 생성"""
        print(f"QKD 데이터셋 생성 시작 (샘플 수: {n_samples})")
        
        # 입력 조합 생성 (매번 다른 랜덤 조합을 위해 시드를 None으로 설정)
        input_combinations = self.generate_input_combinations(n_samples, method='random', random_seed=None)
        
        # 각 조합에 대해 최적화 수행
        dataset = []
        failed_count = 0
        
        for i, input_params in enumerate(tqdm(input_combinations, desc="최적화 진행")):
            try:
                # 최적화 수행
                result = self.optimize_parameters(input_params, max_generations)
                
                # 데이터 포인트 생성 (CSV용으로 평면화)
                data_point = {
                    # 입력 파라미터들
                    'L': input_params['L'],
                    'eta_d': input_params['eta_d'],
                    'Y_0': input_params['Y_0'],
                    'e_d': input_params['e_d'],
                    'alpha': input_params['alpha'],
                    'zeta': input_params['zeta'],
                    'e_0': input_params['e_0'],
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
                }
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
        """저장된 데이터셋 로드"""
        if path.endswith('.csv'):
            return pd.read_csv(path)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    def analyze_dataset(self, dataset):
        """데이터셋 분석"""
        print("=== 데이터셋 분석 ===")
        
        # DataFrame인지 리스트인지 확인
        if isinstance(dataset, pd.DataFrame):
            df = dataset
            print(f"총 샘플 수: {len(df)}")
            
            # SKR 분포 분석
            skr_values = df['skr'].values
            print(f"SKR 범위: {min(skr_values):.2e} ~ {max(skr_values):.2e}")
            print(f"SKR 평균: {np.mean(skr_values):.2e}")
            print(f"SKR 중앙값: {np.median(skr_values):.2e}")
                
        else:
            # 기존 리스트 형태
            print(f"총 샘플 수: {len(dataset)}")
            
            # SKR 분포 분석
            skr_values = [d['skr'] for d in dataset]
            print(f"SKR 범위: {min(skr_values):.2e} ~ {max(skr_values):.2e}")
            print(f"SKR 평균: {np.mean(skr_values):.2e}")
            print(f"SKR 중앙값: {np.median(skr_values):.2e}")

if __name__ == "__main__":
    # 데이터 생성기 초기화
    generator = QKDDataGenerator()
    
    # 작은 데이터셋으로 테스트
    print("테스트 데이터셋 생성 중...")
    test_dataset = generator.generate_dataset(
        n_samples=1000,  # 테스트용으로 작은 수
        max_generations=100,  # 빠른 테스트를 위해 세대 수 줄임
        save_path='test_qkd_dataset.csv'
    )
    
    # 데이터셋 분석
    generator.analyze_dataset(test_dataset)
