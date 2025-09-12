import numpy as np
import yaml

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

# 파생 상수
eps = eps_sec/23
beta = np.log(1/eps)

# L은 전역 변수로 설정
L = 110

def normalize_p(vec):
    """벡터를 정규화하는 함수"""
    copy_vec = vec[:].copy()
    sum_vec = np.sum(copy_vec[3:6])
    copy_vec[3:6] /= sum_vec
    return copy_vec

def h(x):
    """이진 엔트로피 함수"""
    return -x * np.log2(x) - (1 - x)*np.log2(1 - x)

def skr_simulator(_, solution, __) :
    """SKR(Secret Key Rate) 계산 함수"""
    sol = normalize_p(solution)
    mu, nu, vac, p_mu, p_nu, p_vac, p_X, q_X = sol

    p_Z = 1 - p_X
    q_Z = 1 - q_X

    if mu <= nu : 
        return -10

    eta = eta_d * 10 ** (-alpha*L/10)

    Q_mu = 1 - (1 - Y_0) * np.exp(-mu * eta)
    Q_nu = 1 - (1 - Y_0) * np.exp(-nu * eta)
    Q_vac = 1 - (1 - Y_0) * np.exp(-vac * eta)

    n_mu_Z = N * p_mu * p_Z * q_Z * Q_mu
    n_nu_Z = N * p_nu * p_Z * q_Z * Q_nu
    n_vac_Z = N * p_vac * p_Z * q_Z * Q_vac

    n_mu_X = N * p_mu * p_X * q_X * Q_mu
    n_nu_X = N * p_nu * p_X * q_X * Q_nu
    n_vac_X = N * p_vac * p_X * q_X * Q_vac

    if (n_mu_Z<0) or (n_nu_Z<0) or (n_vac_Z<0) or (n_mu_X<0) or (n_nu_X<0) or (n_vac_X<0) :
        return -8
    
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