"""
MLP vs FT-Transformer 성능 비교 스크립트
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_mlp import QKDMLPTrainer, set_seed, TRAINING_CONFIG as MLP_CONFIG
from train_fttransformer import FTTransformerTrainer, TRAINING_CONFIG as FT_CONFIG
from test_mlp import build_test_loader, compute_skr_percent_errors, summarize_percent_errors

# ============================================
# ===== 여기서 비교 설정을 변경하세요 =====
# ============================================
COMPARE_BATCH_SIZE = 256  # 비교용 배치 크기
# ============================================

def evaluate_model(trainer, X_test, y_test, model_name, batch_size):
    """모델 평가 (test_mlp의 함수 재사용)"""
    print(f"\n{'='*60}")
    print(f"{model_name} 평가")
    print(f"{'='*60}")
    
    # test_mlp의 build_test_loader 함수 사용
    test_loader = build_test_loader(trainer, X_test, y_test, batch_size)
    
    # 평가
    results = trainer.evaluate(test_loader)
    
    print(f"전체 MSE: {results['overall_mse']:.6e}")
    print("\n파라미터별 오차:")
    print(f"{'Parameter':<12} {'MSE':<15} {'MAE':<15}")
    print("-" * 42)
    
    param_names = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
    for param in param_names:
        mse = results['param_errors'][param]['mse']
        mae = results['param_errors'][param]['mae']
        print(f"{param:<12} {mse:<15.6e} {mae:<15.6e}")
    
    return results

def compare_predictions(results_mlp, results_ft, L, save_path):
    """예측 결과 비교 시각화"""
    param_names = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
    
    # MSE 비교
    mlp_mses = [results_mlp['param_errors'][p]['mse'] for p in param_names]
    ft_mses = [results_ft['param_errors'][p]['mse'] for p in param_names]
    
    # MAE 비교
    mlp_maes = [results_mlp['param_errors'][p]['mae'] for p in param_names]
    ft_maes = [results_ft['param_errors'][p]['mae'] for p in param_names]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    x = np.arange(len(param_names))
    width = 0.35
    
    # MSE 비교
    axes[0].bar(x - width/2, mlp_mses, width, label='MLP', alpha=0.8)
    axes[0].bar(x + width/2, ft_mses, width, label='FT-Transformer', alpha=0.8)
    axes[0].set_ylabel('MSE (log scale)')
    axes[0].set_title(f'Mean Squared Error Comparison (L={L} km)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(param_names, rotation=45)
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # MAE 비교
    axes[1].bar(x - width/2, mlp_maes, width, label='MLP', alpha=0.8)
    axes[1].bar(x + width/2, ft_maes, width, label='FT-Transformer', alpha=0.8)
    axes[1].set_ylabel('MAE (log scale)')
    axes[1].set_title(f'Mean Absolute Error Comparison (L={L} km)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names, rotation=45)
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n비교 그래프가 {save_path}에 저장되었습니다.")

def print_summary(results_mlp, results_ft, L):
    """비교 결과 요약"""
    print("\n" + "="*60)
    print(f"최종 비교 결과 (L={L} km)")
    print("="*60)
    
    param_names = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
    
    print(f"\n{'Parameter':<12} {'Winner':<15} {'Improvement':<15}")
    print("-" * 42)
    
    mlp_wins = 0
    ft_wins = 0
    
    for param in param_names:
        mlp_mse = results_mlp['param_errors'][param]['mse']
        ft_mse = results_ft['param_errors'][param]['mse']
        
        if mlp_mse < ft_mse:
            winner = "MLP"
            improvement = (ft_mse - mlp_mse) / ft_mse * 100
            mlp_wins += 1
        else:
            winner = "FT-Transformer"
            improvement = (mlp_mse - ft_mse) / mlp_mse * 100
            ft_wins += 1
        
        print(f"{param:<12} {winner:<15} {improvement:>6.2f}%")
    
    print("\n" + "="*60)
    print(f"MLP 승리: {mlp_wins}개 파라미터")
    print(f"FT-Transformer 승리: {ft_wins}개 파라미터")
    
    # 전체 MSE 비교
    mlp_overall = results_mlp['overall_mse']
    ft_overall = results_ft['overall_mse']
    
    if mlp_overall < ft_overall:
        print(f"\n전체 MSE 승자: MLP ({(ft_overall-mlp_overall)/ft_overall*100:.2f}% 더 좋음)")
    else:
        print(f"\n전체 MSE 승자: FT-Transformer ({(mlp_overall-ft_overall)/mlp_overall*100:.2f}% 더 좋음)")
    
    # SKR 비교
    mlp_skr_mse = results_mlp['param_errors']['skr']['mse']
    ft_skr_mse = results_ft['param_errors']['skr']['mse']
    
    print(f"\nSKR MSE 비교:")
    print(f"  MLP:            {mlp_skr_mse:.6e}")
    print(f"  FT-Transformer: {ft_skr_mse:.6e}")
    
    if mlp_skr_mse < ft_skr_mse:
        print(f"  → MLP가 {(ft_skr_mse-mlp_skr_mse)/ft_skr_mse*100:.2f}% 더 좋음")
    else:
        print(f"  → FT-Transformer가 {(mlp_skr_mse-ft_skr_mse)/mlp_skr_mse*100:.2f}% 더 좋음")
    
    print("="*60)

def main():
    # 설정값 사용
    L = MLP_CONFIG['L']
    batch_size = COMPARE_BATCH_SIZE
    seed = 42  # 고정 시드
    
    # 경로 설정 (L 값 기반)
    test_csv = f"dataset/test_L{L}.csv"
    mlp_epochs = MLP_CONFIG['epochs']
    mlp_batch_size = MLP_CONFIG['batch_size']
    ft_epochs = FT_CONFIG['epochs']
    ft_batch_size = FT_CONFIG['batch_size']
    mlp_model = f"qkd_mlp_L{L}_E{mlp_epochs}_B{mlp_batch_size}.pth"
    ft_model = f"qkd_fttransformer_L{L}_E{ft_epochs}_B{ft_batch_size}.pth"
    output_path = f"model_comparison_L{L}.png"
    
    print("="*80)
    print(f"QKD 모델 성능 비교: MLP vs FT-Transformer (L={L} km)")
    print("="*80)
    
    # 재현 가능한 결과를 위한 시드 고정
    set_seed(seed)
    
    # 테스트 데이터 로드
    print(f"\n테스트 데이터 로드: {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"테스트 데이터: {len(df)} 샘플")
    
    input_columns = ['eta_d', 'Y_0', 'e_d', 'alpha', 'zeta', 'eps_sec', 'eps_cor', 'N']
    output_columns = ['mu', 'nu', 'vac', 'p_mu', 'p_nu', 'p_vac', 'p_X', 'q_X', 'skr']
    
    X_test = df[input_columns].values
    y_test = df[output_columns].values
    
    # MLP 모델 평가
    print("\n[1/2] MLP 모델 로드 및 평가...")
    try:
        mlp_trainer = QKDMLPTrainer()
        mlp_trainer.load_model(mlp_model)
        results_mlp = evaluate_model(mlp_trainer, X_test, y_test, "MLP", batch_size)
    except FileNotFoundError:
        print(f"오류: {mlp_model} 파일을 찾을 수 없습니다.")
        print(f"먼저 'python train_mlp.py'를 실행하여 MLP 모델을 학습시키세요.")
        return
    
    # FT-Transformer 모델 평가
    print("\n[2/2] FT-Transformer 모델 로드 및 평가...")
    try:
        ft_trainer = FTTransformerTrainer()
        ft_trainer.load_model(ft_model)
        results_ft = evaluate_model(ft_trainer, X_test, y_test, "FT-Transformer", batch_size)
    except FileNotFoundError:
        print(f"오류: {ft_model} 파일을 찾을 수 없습니다.")
        print(f"먼저 'python train_fttransformer.py'를 실행하여 FT-Transformer 모델을 학습시키세요.")
        return
    
    # 비교 시각화
    compare_predictions(results_mlp, results_ft, L, output_path)
    
    # 요약 출력
    print_summary(results_mlp, results_ft, L)
    
    print("\n" + "="*80)
    print("비교 완료!")
    print("="*80)

if __name__ == "__main__":
    main()
