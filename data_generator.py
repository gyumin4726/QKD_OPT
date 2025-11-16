import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
from ga_final import define_ga
from simulator import skr_simulator

class QKDDataGenerator:
    def __init__(self, L=None):
        """QKD 학습 데이터 생성기
        
        Args:
            L: 고정할 거리 값 (km). 필수 파라미터입니다.
        
        Raises:
            ValueError: L이 지정되지 않은 경우
        """
        # L 값 검증 (필수 파라미터)
        if L is None:
            raise ValueError("L 값은 필수입니다. L 파라미터를 지정해주세요 (예: L=10)")
        
        # L 값 설정
        self.fixed_L = L
        
        # 기본 파라미터 범위 설정
        # 배경 오류율(e_0)은 0.5로 고정, 변수 관 관계는 대표님이 주신다고 하심, L별로 따라 모델을 만들어서 학습
        self.param_ranges = {
            'eta_d': (0.02, 0.08),           # 탐지기 효율 (2-8%, 기본값 4.5%)
            'Y_0': (1e-7, 1e-5),             # 다크 카운트율 (기본값 1.7e-6)
            'e_d': (0.02, 0.05),             # 오정렬률 (2-5%, 기본값 3.3%)
            'alpha': (0.18, 0.24),           # 광섬유 감쇠 계수 (기본값 0.21)
            'zeta': (1.1, 1.4),              # 오류 정정 효율 (기본값 1.22)
            'eps_sec': (1e-12, 1e-8),        # 보안 파라미터 (기본값 1e-10)
            'eps_cor': (1e-18, 1e-12),       # 정확성 파라미터 (기본값 1e-15)
            'N': (1e9, 1e11)                 # 광 펄스 수 (기본값 1e10)
        }
        
        # 고정 파라미터: e_0(배경 오류율)는 항상 0.5로 고정
        self.fixed_params = {
            'e_0': 0.5                       # 배경 오류율 고정값
        }
        
        print(f"L={self.fixed_L} km로 고정하여 데이터셋 생성")
    
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
                # 고정 파라미터 먼저 추가
                for param, value in self.fixed_params.items():
                    combo[param] = value
                
                # L은 고정값 사용
                combo['L'] = self.fixed_L
                
                # 변수 파라미터 샘플링
                for param, (min_val, max_val) in self.param_ranges.items():
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
            
            # L은 고정값 사용
            L = self.fixed_L
            
            for eta_d in np.linspace(*self.param_ranges['eta_d'], n_per_param):
                for Y_0 in np.logspace(
                    np.log10(self.param_ranges['Y_0'][0]),
                    np.log10(self.param_ranges['Y_0'][1]),
                    n_per_param
                ):
                    for e_d in np.linspace(*self.param_ranges['e_d'], n_per_param):
                        for alpha in np.linspace(*self.param_ranges['alpha'], n_per_param):
                            for zeta in np.linspace(*self.param_ranges['zeta'], n_per_param):
                                # e_0는 고정값 사용
                                e_0 = self.fixed_params['e_0']
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
        
        # vac 최적화 모드 설정 (vac도 최적화하도록 True로 설정)
        OPTIMIZE_VAC = True
        
        # ga_final.py의 전역 OPTIMIZE_VAC 변수도 설정 (define_ga 함수가 이를 참조함)
        import ga_final
        ga_final.OPTIMIZE_VAC = True
        
        # PyGAD용 래퍼 함수 - 개별 파라미터를 직접 전달
        def skr_fitness_wrapper(ga_instance, solution, solution_idx):
            if OPTIMIZE_VAC:
                # vac도 최적화: 8개 유전자 그대로 사용
                return skr_simulator(ga_instance, solution, solution_idx, **input_params)
            else:
                # vac=0 고정: 7개 유전자를 8개로 확장
                # solution은 7개: mu, nu, p_mu, p_nu, p_vac, p_X, q_X
                # vac=0을 인덱스 2에 삽입하여 8개로 만듦
                full_solution = np.insert(solution, 2, 0.0)
                return skr_simulator(ga_instance, full_solution, solution_idx, **input_params)
        
        # 최적화된 GA 설정 (README.md에서 가져온 설정)
        ga = define_ga(
            co_type='scattered',
            mu_type='adaptive',
            sel_type='tournament',
            gen=max_generations,
            num_parents_mating=41,
            sol_per_pop=188,
            keep_parents=40,
            keep_elitism=8,
            K_tournament=25,
            crossover_probability=0.549428712068568,
            mutation_probability=None,  # adaptive mutation 사용
            mutation_percent_genes=[0.7, 0.2],
            random_seed=42,
            fitness_func=skr_fitness_wrapper  # 래퍼 함수 사용
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
    
    def generate_dataset(self, n_samples=1000, max_generations=50, save_path=None):
        """전체 데이터셋 생성
        
        Args:
            n_samples: 생성할 샘플 수
            max_generations: GA 최대 세대 수
            save_path: 저장 경로 (None이면 L 값에 따라 자동 생성)
        """
        # 저장 경로가 지정되지 않았으면 L 값에 따라 자동 생성
        if save_path is None:
            # dataset 폴더가 없으면 생성
            dataset_dir = 'dataset'
            os.makedirs(dataset_dir, exist_ok=True)
            save_path = os.path.join(dataset_dir, f'qkd_dataset_L{self.fixed_L}.csv')
        else:
            # 지정된 경로도 dataset 폴더 안에 저장 (경로가 이미 지정된 경우는 그대로 사용)
            if not os.path.isabs(save_path) and not save_path.startswith('dataset/'):
                dataset_dir = 'dataset'
                os.makedirs(dataset_dir, exist_ok=True)
                save_path = os.path.join(dataset_dir, save_path)
        
        print(f"QKD 데이터셋 생성 시작 (L={self.fixed_L} km 고정, 샘플 수: {n_samples})")
        
        # 입력 조합 생성 (재현 가능성을 위해 시드를 42로 고정)
        input_combinations = self.generate_input_combinations(n_samples, method='random', random_seed=42)
        
        # 각 조합에 대해 최적화 수행
        dataset = []
        failed_count = 0
        
        for i, input_params in enumerate(tqdm(input_combinations, desc="최적화 진행")):
            try:
                # 최적화 수행
                result = self.optimize_parameters(input_params, max_generations)
                
                # 데이터 포인트 생성 (CSV용으로 평면화)
                # L과 e_0는 고정값이므로 CSV에 저장하지 않음 (파일명에 L 값이 포함되어 있음)
                data_point = {
                    # 입력 파라미터들 (L과 e_0 제외 - 고정값)
                    'eta_d': input_params['eta_d'],
                    'Y_0': input_params['Y_0'],
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
    # L 값을 지정하여 데이터 생성기 초기화 (L은 필수 파라미터)
    generator = QKDDataGenerator(L=20)  # L 고정 (필요 시 다른 거리로 변경)
    
    # 작은 데이터셋으로 테스트
    print("테스트 데이터셋 생성 중...")
    test_dataset = generator.generate_dataset(
        n_samples=100000,  # 테스트용으로 작은 수
        max_generations=100,  # 빠른 테스트를 위해 세대 수 줄임
        save_path=None  # None이면 자동으로 L 값이 파일명에 포함됨 (qkd_dataset_L10.csv)
    )
    
    # 데이터셋 분석
    generator.analyze_dataset(test_dataset)
