# QKD Parameter Optimization

양자 키 분배(QKD) 시스템의 최적 파라미터 탐색 및 예측 모델 학습 프로젝트.

## 개요

QKD 환경 변수에 따른 최적 파라미터(mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X)와 SKR(Secure Key Rate)을 유전 알고리즘(GA)으로 탐색하고, 신경망 모델로 예측.

## 시스템 구성

### Core Modules
- `simulator.py` - SKR 계산 엔진 (에러 코드: -1 ~ -12)
- `calculate_skr.py` - 독립 실행형 SKR 계산 유틸리티

### Optimization
- `ga_final.py` - GA 최적화 + Optuna 하이퍼파라미터 튜닝
- `ga_crosscheck.py` - 최적화된 GA 파라미터로 실행하여 검증

### Data Pipeline
```
1. data_generator.py     → raw_dataset_L{L}.csv
2. clean_dataset.py      → cleaned_dataset_L{L}.csv
3. data_split.py         → train_L{L}.csv, test_L{L}.csv
4. train_*.py → 모델 학습
5. test_*.py  → 모델 평가
```

### Models
- **FT-Transformer** : Feature Tokenizer + Transformer 기반 tabular 모델
- **MLP**: 512-256 은닉층 구조의 기본 신경망

## 사용법

### 1. 데이터셋 생성

```bash
# data_generator.py 상단 설정
DEFAULT_L = 100                  # 거리 (km)
OUTPUT_FILENAME = f'raw_dataset_L{DEFAULT_L}.csv'

python data_generator.py
```

### 2. 데이터 정제

```bash
# clean_dataset.py 상단 설정
L = 100

python clean_dataset.py
```

### 3. 데이터 분할

```bash
# data_split.py 상단 설정
L = 100
TEST_SIZE = 0.2

python data_split.py
```

### 4. 모델 학습

```bash
# train_fttransformer.py 상단 설정
L = 100
EPOCHS = 500
BATCH_SIZE = 64

python train_fttransformer.py
```

### 5. 모델 평가

```bash
# test_fttransformer.py 상단 설정
L = 100 (모델 추적용)
EPOCHS = 500 (모델 추적용)
BATCH_SIZE = 64 (모델 추적용)

python test_fttransformer.py
```

## 설정 관리

모든 설정은 각 파일 상단에서 관리. 주요 변수:

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `L` | 거리 (km) | 100 |
| `EPOCHS` | 훈련 에포크 | 500 (FT-Transformer), 200 (MLP) |
| `BATCH_SIZE` | 배치 크기 | 64 (FT-Transformer), 128 (MLP) |
| `OPTIMIZER` | 옵티마이저 | Adam |
| `LEARNING_RATE` | 학습률 | 0.0005 (FT-Transformer), 0.001 (MLP) |

**중요**: 파이프라인을 연결하기 위해서는 `L` 값을 모든 파일에서 동일하게 설정.

## 데이터 형식

### 입력 변수 (7개)
- `eta_d` - 검출기 효율
- `e_d` - 오정렬률
- `alpha` - 광섬유 감쇠 계수
- `zeta` - 오류 정정 효율
- `eps_sec` - 보안 파라미터
- `eps_cor` - 정확성 파라미터
- `N` - 광 펄스 수

**고정값**: `Y_0` (암계수율), `e_0` (배경 오류율), `L` (거리)

### 출력 변수 (9개)
- `mu`, `nu`, `vac` - 신호 강도
- `p_mu`, `p_nu`, `p_vac` - 신호 확률
- `p_X`, `q_X` - 기저 선택 확률
- `skr` - Secure Key Rate

## 디렉토리 구조

```
QKD_OPT/
├── dataset/                    # 데이터셋 (자동 생성)
│   ├── raw_dataset_L{L}.csv
│   ├── cleaned_dataset_L{L}.csv
│   ├── train_L{L}.csv
│   └── test_L{L}.csv
├── simulator.py                # SKR 계산 코어
├── calculate_skr.py            # SKR 계산 유틸리티
├── ga_final.py                 # GA 최적화
├── ga_crosscheck.py            # GA 검증
├── data_generator.py           # 데이터 생성
├── clean_dataset.py            # 데이터 정제
├── data_split.py               # 데이터 분할
├── train_fttransformer.py      # FT-Transformer 학습
├── test_fttransformer.py       # FT-Transformer 평가
├── train_mlp.py                # MLP 학습 (레거시)
├── test_mlp.py                 # MLP 평가 (레거시)
└── qkd_*transformer_*.pth      # 학습된 모델 (자동 생성)
```

## 설치

### 1. 가상환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

## SKR 에러 코드

| 코드 | 설명 |
|------|------|
| -1 | NaN 또는 Inf SKR |
| -2 | 생성 키 길이 범위 오류 (length > N or < 0) |
| -3 | phi_1_Z_U 범위 오류 (> 0.5 or < 0) |
| -4 | S_1_Z_L 또는 S_1_X_L 음수 |
| -5 | 보정된 단일 광자 이벤트 수 음수 |
| -6 | 예상 단일 광자 이벤트 수 음수 |
| -7 | Bound 계산 결과 음수 |
| -8 | 오류 이벤트 수 음수 (m_mu_Z, m_nu_Z, m_nu_X) |
| -9 | 이벤트 수 음수 (n_mu_Z, n_nu_Z, n_vac_Z 등) |
| -10 | 강도 파라미터 크기 오류 (mu <= nu) |
| -11 | 강도 파라미터 크기 오류 (nu <= vac) |
| -12 | 확률 파라미터 크기 오류 (p_mu <= p_nu or p_nu <= p_vac) |

## 참고사항

- 모든 설정은 각 Python 파일 상단에서 직접 수정
- 로그 변환: `eps_sec`, `eps_cor`, `N`, `skr`
- 정규화: MinMaxScaler (입력/출력 모두)
- FT-Transformer 권장 (MLP 대비 성능 우수, 학습 시간 다소 증가)

## 참고문헌

1. Yin, H. L., Zhou, M. G., Gu, J., Xie, Y. M., Lu, Y. S., & Chen, Z. B. (2020). Tight security bounds for decoy-state quantum key distribution. *Scientific Reports*, *10*(1), 14312. https://doi.org/10.1038/s41598-020-71107-6

2. Wang, W., & Lo, H. K. (2019). Machine Learning for Optimal Parameter Prediction in Quantum Key Distribution. *arXiv preprint arXiv:1812.07724*. https://doi.org/10.48550/arXiv.1812.07724

3. Mahmud, I., & Abdelhadi, A. (2025). Artificial Intelligence in Quantum Communications: A Comprehensive Survey. *IEEE Access*, *13*, 121175-121210. https://doi.org/10.1109/ACCESS.2025.3585799

4. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *Advances in Neural Information Processing Systems*, *34*, 18932-18943. https://doi.org/10.48550/arXiv.2106.11959

5. Somepalli, G., Schwarzschild, A., Goldblum, M., Bruss, C. B., & Goldstein, T. (2021). SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training. *arXiv preprint arXiv:2106.01342*. https://doi.org/10.48550/arXiv.2106.01342

6. Decker, T., Gallezot, M., Kerstan, S. F., Paesano, A., Ginter, A., & Wormsbecher, W. (2025). Quantum key distribution as a quantum machine learning task. *npj Quantum Information*, *11*(1), 140. https://doi.org/10.1038/s41534-025-01088-9

7. Long, N. K., Malaney, R., & Grant, K. J. (2023). A Survey of Machine Learning Assisted Continuous-Variable Quantum Key Distribution. *Information*, *14*(10), 553. https://doi.org/10.3390/info14100553