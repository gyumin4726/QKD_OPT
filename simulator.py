import numpy as np
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class QKDSimulatorConfig:
    """QKD 시뮬레이터 설정 클래스"""
    # Detection parameters
    eta_d: float
    Y_0: float
    e_d: float
    
    # Fiber parameters
    alpha: float
    
    # Error correction parameters
    zeta: float
    e_0: float
    
    # Security parameters
    eps_sec: float
    eps_cor: float
    
    # System parameters
    N: float
    Lambda: float
    L: float = 110  # 기본값 설정
    
    def __post_init__(self):
        """파생 상수 계산 (더 이상 필요 없음)"""
        pass

    @classmethod
    def from_yaml(cls, config_path: str = 'config/config.yaml') -> 'QKDSimulatorConfig':
        """YAML 파일에서 설정을 로드하는 클래스 메서드"""
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return cls(
            eta_d=float(config['detection']['eta_d']),
            Y_0=float(config['detection']['Y_0']),
            e_d=float(config['detection']['e_d']),
            alpha=float(config['fiber']['alpha']),
            zeta=float(config['error_correction']['zeta']),
            e_0=float(config['error_correction']['e_0']),
            eps_sec=float(config['security']['eps_sec']),
            eps_cor=float(config['security']['eps_cor']),
            N=float(config['system']['N']),
            Lambda=config['system']['Lambda'],
            L=110  # 기본값
        )

# 전역 변수 제거됨 - 이제 QKDSimulatorConfig 클래스 사용

def normalize_p(vec):
    """벡터를 정규화하는 함수"""
    copy_vec = vec[:].copy()
    sum_vec = np.sum(copy_vec[3:6])
    copy_vec[3:6] /= sum_vec
    return copy_vec

def h(x):
    """이진 엔트로피 함수"""
    return -x * np.log2(x) - (1 - x)*np.log2(1 - x)

def skr_simulator(_, solution, __, config=None, **kwargs):
    """SKR(Secret Key Rate) 계산 함수
    
    Args:
        _: 첫 번째 매개변수 (사용되지 않음)
        solution: 최적화할 파라미터 벡터
        __: 세 번째 매개변수 (사용되지 않음)
        config: QKDSimulatorConfig 객체 (선택사항)
        **kwargs: 개별 파라미터들 (L, eta_d, Y_0, e_d, alpha, zeta, e_0, eps_sec, eps_cor, N, Lambda)
    """
    sol = normalize_p(solution)
    mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X = sol

    p_Z = 1 - p_X
    q_Z = 1 - q_X

    if mu <= nu : 
        return -10
    
    if nu <= vac:
        return -11  # nu > vac 제약 위반
    
    if p_mu <= p_nu or p_nu <= p_vac:
        return -12  # p_mu > p_nu > p_vac 제약 위반

    # config가 제공되면 config에서 값 추출, 아니면 kwargs에서 추출
    if config is not None:
        eta_d = config.eta_d
        alpha = config.alpha
        L = config.L
        Y_0 = config.Y_0
        e_d = config.e_d
        e_0 = config.e_0
        N = config.N
        zeta = config.zeta
        eps_sec = config.eps_sec
        eps_cor = config.eps_cor
        Lambda = config.Lambda
        
        # 파생 상수 계산
        eps = eps_sec / 23
        beta = np.log(1 / eps)
    else:
        # kwargs에서 개별 파라미터 추출
        eta_d = kwargs.get('eta_d')
        alpha = kwargs.get('alpha')
        L = kwargs.get('L')
        Y_0 = kwargs.get('Y_0')
        e_d = kwargs.get('e_d')
        e_0 = kwargs.get('e_0')
        N = kwargs.get('N')
        zeta = kwargs.get('zeta')
        eps_sec = kwargs.get('eps_sec')
        eps_cor = kwargs.get('eps_cor')
        Lambda = kwargs.get('Lambda')
        
        # 파생 상수 계산
        eps = eps_sec / 23
        beta = np.log(1 / eps)

    eta = eta_d * 10 ** (-alpha*L/10)

    Q_mu = 1 - (1 - Y_0) * np.exp(-mu * eta)
    Q_nu = 1 - (1 - Y_0) * np.exp(-nu * eta)
    Q_vac = 1 - (1 - Y_0) * np.exp(-vac * eta)

    n_mu_Z = N * p_mu * p_Z * q_Z * Q_mu
    n_nu_Z = N * p_nu * p_Z * q_Z * Q_nu
    # n_vac_Z = N * p_vac * p_Z * q_Z * Q_vac
    n_vac_Z = N * p_vac * q_Z * Q_vac 

    n_mu_X = N * p_mu * p_X * q_X * Q_mu
    n_nu_X = N * p_nu * p_X * q_X * Q_nu
    #n_vac_X = N * p_vac * p_X * q_X * Q_vac
    n_vac_X = N * p_vac * q_X * Q_vac

    if (n_mu_Z<0) or (n_nu_Z<0) or (n_vac_Z<0) or (n_mu_X<0) or (n_nu_X<0) or (n_vac_X<0) :
        return -9
    
    m_mu_Z = N * p_mu * p_Z * q_Z * (e_d * Q_mu + (e_0 - e_d)*Y_0)
    m_nu_Z = N * p_nu * p_Z * q_Z * (e_d * Q_nu + (e_0 - e_d)*Y_0)
    m_nu_X = N * p_nu * p_X * q_X * (e_d * Q_nu + (e_0 - e_d)*Y_0)

    if (m_mu_Z<0) or (m_nu_Z<0) or (m_nu_X<0) :
        return -8
    
    # Z-basis lower bound
    n_0_z_L_ex = n_vac_Z - beta/2-np.sqrt(2*beta*n_vac_Z+beta**2/4)
    n_nu_z_L_ex = n_nu_Z - beta/2-np.sqrt(2*beta*n_nu_Z+beta**2/4)

    # Z-basis upper bound
    n_mu_z_U_ex = n_mu_Z + beta+np.sqrt(2*beta*n_mu_Z+beta**2)
    n_0_z_U_ex = n_vac_Z + beta+np.sqrt(2*beta*n_vac_Z+beta**2)

    # X-basis lower bound
    n_0_x_L_ex = n_vac_X - beta/2-np.sqrt(2*beta*n_vac_X+beta**2/4)
    n_nu_x_L_ex = n_nu_X - beta/2-np.sqrt(2*beta*n_nu_X+beta**2/4)                 

    # X-basis upper bound
    n_mu_x_U_ex = n_mu_X + beta+np.sqrt(2*beta*n_mu_X+beta**2)
    n_0_x_U_ex = n_vac_X + beta+np.sqrt(2*beta*n_vac_X+beta**2)

    # error upper bound
    m_nu_x_U_ex = m_nu_X + beta+np.sqrt(2*beta*m_nu_X+beta**2)

    if (n_0_z_L_ex<0) or (n_nu_z_L_ex<0) or (n_mu_z_U_ex<0) or (n_0_z_U_ex<0) or (n_0_x_L_ex<0) or (n_nu_x_L_ex<0) or (n_mu_x_U_ex<0) or (n_0_x_U_ex<0) or (m_nu_x_U_ex<0) :
        return -7
    
    # lower bound on the expected number of vacuum event
    S_0_Z_L_ex = (np.exp(-mu)*p_mu+np.exp(-nu)*p_nu)*p_Z*n_0_z_L_ex/p_vac
    # lower bound on the expected number of single photon event
    S_1_Z_L_ex = (mu**2*np.exp(-mu)*p_mu+mu*nu*np.exp(-nu)*p_nu)/(mu*nu-nu**2)*(np.exp(nu)*n_nu_z_L_ex/p_nu-nu**2/mu**2*np.exp(mu)*n_mu_z_U_ex/p_mu-(mu**2-nu**2)/mu**2*p_Z*n_0_z_U_ex/p_vac)
    # lower bound on the expected number of single-photon events
    S_1_X_L_ex = (mu**2*np.exp(-mu)*p_mu+mu*nu*np.exp(-nu)*p_nu)/(mu*nu-nu**2)*(np.exp(nu)*n_nu_x_L_ex/p_nu-nu**2/mu**2*np.exp(mu)*n_mu_x_U_ex/p_mu-(mu**2-nu**2)/mu**2*p_X*n_0_x_U_ex/p_vac)
    # upper bound on the expected number of bit error
    T_1_X_U_ex = ((mu*np.exp(-mu)*p_mu+nu*np.exp(-nu)*p_nu)/nu)*(np.exp(nu)*m_nu_x_U_ex/p_nu-p_X*n_0_x_L_ex/(2*p_vac))

    if (S_0_Z_L_ex<0)or(S_1_Z_L_ex<0)or(S_1_X_L_ex<0)or(T_1_X_U_ex<0) : 
        return -6

    S_0_Z_L = S_0_Z_L_ex - np.sqrt(2*beta*S_0_Z_L_ex)
    S_1_Z_L = S_1_Z_L_ex - np.sqrt(2*beta*S_1_Z_L_ex)
    S_1_X_L = S_1_X_L_ex - np.sqrt(2*beta*S_1_X_L_ex)
    T_1_X_U = T_1_X_U_ex + beta/2+np.sqrt(2*beta*T_1_X_U_ex+beta**2/4)

    if (S_0_Z_L<0)or(S_1_Z_L<0)or(S_1_X_L<0)or(T_1_X_U<0) : 
        return -5

    n = S_1_Z_L
    k = S_1_X_L
    Lambda = T_1_X_U/S_1_X_L

    if (n < 0) or (k < 0) : 
        return -4

    A = np.max([n,k])
    G = (n+k)/(n*k) * np.log((n+k) / (2*np.pi*n*k*Lambda*(1-Lambda)*eps**2))

    gamma_U = (((1 - 2 * Lambda)*A*G)/(n+k) + np.sqrt(A**2*G**2/(n+k)**2 + 4*Lambda*(1-Lambda)*G))/ (2 + 2*A**2*G/(n + k)**2)

    phi_1_Z_U =  Lambda + gamma_U
    if (phi_1_Z_U > 0.5) or (phi_1_Z_U <0):
        return -3

    # 생성된 키 길이 계산
    n_Z = n_mu_Z + n_nu_Z
    E_Z = (m_mu_Z + m_nu_Z)/n_Z

    lambda_ec = n_Z * zeta * h(E_Z)

    length = S_0_Z_L + S_1_Z_L * (1 - h(phi_1_Z_U)) - lambda_ec - np.log2(2/eps_cor) - 6*np.log2(23/eps_sec)

    if (length > N) or (length < 0) : 
        return -2

    SKR = length/N

    if np.isnan(SKR) or np.isinf(SKR):
        return -1

    return SKR