import numpy as np
from scipy.special import expit

def column_one(num):
    value = np.ones((num, 1))
    return value

def generate_S0(n0):
    p = 6
    S0 = np.random.normal(size=n0*p)
    S0 = np.reshape(S0, (n0, p))
    return S0

def generate_Zt(n0=None, prob=False):
    p = 0.61063
    if prob:
        return p * np.ones(n0)
    else:
        Zt = np.random.binomial(n=1, p=p, size=n0)    
        return Zt

def generate_At(St, Zt, n0=None, prob=False):
    coef = np.array([
        -0.21679366086128768, 
        9.461230469227252e-05,-9.004337164002584e-06,1.4601568148343578e-07,1.8033063227029235e-08,-1.6847592785865295e-07,1.0356903409672021e-06,
        0.2575423324576441
    ])
    SZt = np.hstack([column_one(St.shape[0]), St, Zt.reshape(-1, 1)])
    value = SZt.dot(coef)
    p = expit(value)
    if prob:
        return p
    else:
        At = np.random.binomial(n=1, p=p)
        return At

def generate_Rt(St,Zt,At,n0=None):
    coef = np.array([
        -0.3148796663285785,
        3.7362269306654226e-05,6.395125785757253e-05,1.107007369313043e-06,1.705700756342308e-08,1.078972922490126e-06,4.098567692199781e-06,0.05633737254684471,-0.025308132010646314
    ])
    SZAt = np.hstack([column_one(St.shape[0]), St, Zt.reshape(-1, 1), At.reshape(-1, 1)])
    Rt = SZAt.dot(coef)
    Rt = np.random.normal(loc=Rt)
    return Rt

def generate_Stplus1(St,Zt,At,n0=None):
    intercept = np.array([
        -0.011556835534910131, 
        -0.011474520838004012, 
        0.0810579239284889,
        0.00637692419451814, 
        7.289454957740665e-05,
        0.00013026905956357987, 
    ])
    coef = np.array([
        [-5.2824368324806765e-05,-8.745430278583439e-06,1.548641280746848e-07,-8.714708435775762e-10,-2.2367912523677978e-07,8.044968255960903e-07,-0.005008406537604402,0.0253065899289611],
        [2.432285880656101e-06,9.515997635120526e-07,5.966708147563058e-08,-6.449195689639741e-10,4.267272684903107e-08,1.069190317978605e-07,0.003013753559905255,0.018742268780868192],
        [5.760922232353478e-07,3.447744761301239e-07,-1.0027097941132044e-06,-2.5611759932397755e-09,-3.193197552594171e-08,3.606661681315803e-08,-0.0007759974127952578,-0.009022790515342256],
        [-2.393475010507096e-07,1.7522652404624427e-07,-4.536356625943593e-08,-6.619371834048992e-09,1.6172902985236706e-08,-6.502068372427407e-09,0.0002631234727091995,0.00018217485792293962],
        [1.035827394595801e-07,1.5622501880956704e-07,1.091809911879071e-09,4.102550358464025e-12,-2.6338950792581974e-09,1.1061684342770932e-09,-0.0002171659768597887,0.0001213744666941983],
        [1.26655867600341e-07,1.2764397854565894e-07,1.2578794473769423e-09,-2.8857663522168356e-12,-4.204199448080716e-09,3.4813742419244678e-09,-0.00025021705330378696,9.441696666747325e-05],
    ])
    coef = np.hstack([intercept.reshape(-1, 1), coef]).T
    SZAt = np.hstack([column_one(St.shape[0]), St, Zt.reshape(-1, 1), At.reshape(-1, 1)])
    value = St + np.matmul(SZAt, coef)
    # noise = 0
    noise = np.random.multivariate_normal(mean=np.zeros(value.shape[1]), cov=np.identity(value.shape[1]), size=value.shape[0])
    Stplus1 = value + noise
    return Stplus1

def realdata_target_Pi(St, prob=True):
    coef = np.array([
        -0.21679366086128768 + 0.1572630744686112, 
        9.461230469227252e-05,-9.004337164002584e-06,1.4601568148343578e-07,1.8033063227029235e-08,-1.6847592785865295e-07,1.0356903409672021e-06,
    ])
    scale = -0.8
    if len(St.shape) == 1:
        SZt = np.concatenate([np.ones(1), St])
        value = SZt.dot(coef)
        p = expit(scale*value)
        p = p.flatten()[0]
    else:
        SZt = np.hstack([np.ones((St.shape[0], 1)), St])
        value = SZt.dot(coef)
        p = expit(scale*value)
        p = p.flatten()
    if prob:
        return p
    else:
        At = np.random.binomial(n=1, p=p)
        return At

COMPUTE_TRUE_VALUE = False
if COMPUTE_TRUE_VALUE:
    # np.random.seed(525)
    np.random.seed(319)
    n0=4*10**7 # the number of trajectories in MC approximation. The larger the better.
    T_burn=20
    # the number of stages
    T = 30
    gamma=0.9

    St=generate_S0(n0)
    for i in range(T_burn):
        Zt=generate_Zt(n0=n0)
        At=generate_At(St, Zt)
        Rt=generate_Rt(St,Zt,At,n0)
        St=generate_Stplus1(St,Zt,At,n0)
    S0_burn = St
    print("Burn-in step done!")

    St = np.copy(S0_burn)
    c_value = np.ones(n0)
    pz_value = np.ones(n0)
    eta_pi_mc = np.zeros(n0)
    for i in range(T):
        Zt=generate_Zt(n0=n0)
        pZ1 = generate_Zt(n0=n0, prob=True)
        pz_value = pz_value * (Zt * pZ1 + (1 - Zt) * (1.0 - pZ1))
        pi=realdata_target_Pi(St)
        pA = generate_At(St, Zt, prob=True)
        pA1Z1 = generate_At(St, np.ones(n0), prob=True)
        pA1Z0 = generate_At(St, np.zeros(n0), prob=True)
        c_value = c_value * ((Zt * (pi - pA1Z0) + (1 - Zt) * (pA1Z1 - pi)) / (pA1Z1 - pA1Z0))
        At=generate_At(St, Zt)
        Rt=generate_Rt(St, Zt, At, n0)
        Rt_new = Rt * (c_value / pz_value)
        St=generate_Stplus1(St, Zt, At, n0)
        eta_pi_mc += gamma**(i) * Rt_new
    print("True value of pi: {} with std {} and se {}".format(eta_pi_mc.mean(), eta_pi_mc.std(), eta_pi_mc.std()/np.sqrt(n0)))

    # St = np.copy(S0_burn)
    # eta_b_mc = np.zeros(n0)
    # for i in range(T):
    #     Zt=generate_Zt(n0=n0)
    #     At=generate_At(St, Zt)
    #     Rt=generate_Rt(St, Zt, At, n0)
    #     St=generate_Stplus1(St, Zt, At, n0)
    #     eta_b_mc += gamma**(i) * Rt
    # print("True value of pi: {} with std {} and se {}".format(eta_b_mc.mean(), eta_b_mc.std(), eta_b_mc.std()/np.sqrt(n0)))

# True value of pi: -2.72691877295589 with std 457.21826656380824 and se 0.0722925555087821
# True value of pi: -2.579754666030088 with std 392.9126023530436 and se 0.06212493724098258