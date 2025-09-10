# QKD 파라미터 최적화 결과

## L=100에서의 최적화 결과

### 최적화된 GA 파라미터
```python
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
```

### 최적화 결과
- **최적 SKR 값**: `9.883606e-06`
- **최적 솔루션**: `[0.879389, 0.181233, 0.136371, 0.013206, 0.911127, 0.094369, 0.10079, 0.086119]`

### 솔루션 파라미터 설명
- `mu`: 0.879389 (강도 파라미터)
- `nu`: 0.181233 (약한 강도 파라미터)  
- `vac`: 0.136371 (진공 상태 파라미터)
- `p_mu`: 0.013206 (mu 상태 확률)
- `p_nu`: 0.911127 (nu 상태 확률)
- `p_vac`: 0.094369 (진공 상태 확률)
- `p_X`: 0.10079 (X 기저 확률)
- `q_X`: 0.086119 (X 기저 확률)

이 파라미터 조합으로 L=100km 광섬유에서 최적의 SKR 성능을 달성했습니다.
