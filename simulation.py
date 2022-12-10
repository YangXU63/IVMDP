import joblib
import numpy as np
import random
import time
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from numpy.ma.core import exp

from IVRL_TE_est import IVRL_TE_est
from DRL_NUC_TE_est import DRL_NUC_TE_est

REAL_DATA = True
if REAL_DATA:
    from model_real import realdata_target_Pi as Pi
    from model_real import generate_S0, generate_At, generate_Zt, generate_Rt, generate_Stplus1
    p = 6
else:
    from policy import trivial_Pi as Pi
    from policy import behavior_Pi as Pi
    from policy import target_Pi as Pi
    from model1 import generate_S0, generate_Ut, generate_At, generate_Zt, generate_Rt, generate_Stplus1
    p = 2

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# generate simulation dataset
# the number of samples
n = 10**3

gamma = 0.9
domain_At = np.array([0, 1])
domain_Zt = np.array([0, 1])
max_iter = 1000
epsilon = 5e-4

REP = 100
est_IVDM = np.zeros((REP, 10))
est_IVMDP = np.zeros((REP, 10))
est_DM = np.zeros((REP, 10))
est_MIS = np.zeros((REP, 10))
est_DRL = np.zeros((REP, 10))

np.random.seed(525)

n_all = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

for rep in range(REP):
  print("rep=", rep, ":\n")
  for j in range(len(n_all)):

    # generate MDP
    n = n_all[j]

    # the number of stages
    T = 120

    S0 = generate_S0(n)
    MDP = np.copy(S0)
    St = S0
    for i in range(T):
        if REAL_DATA:
            Zt = generate_Zt(n)
            At = generate_At(St, Zt, n)
            Rt = generate_Rt(St, Zt, At, n)
            Stplus1 = generate_Stplus1(St, Zt, At, n)
        else:
            Ut = generate_Ut(n)
            Zt = generate_Zt(St, n)
            At = generate_At(St, Zt, Ut, n)
            Rt = generate_Rt(St, At, Ut, n)
            Stplus1 = generate_Stplus1(St, At, Ut, n)
        MDP = np.hstack(
            (MDP, Zt.reshape(-1, 1), At.reshape(-1, 1), Rt.reshape(-1, 1), Stplus1)
        )
        St = Stplus1

    MDP = MDP[:, (20*(p+3)):(120*(p+3)+p)]
    T = 100

    est_IVDM[rep, j], est_IVMDP[rep, j] = IVRL_TE_est(
        MDP, p, n, T, Pi, gamma, domain_At, domain_Zt, max_iter=1000, epsilon=1e-3, 
        ndim=10,
    )
    est_MIS[rep, j], est_DM[rep, j], est_DRL[rep, j] = DRL_NUC_TE_est(
        MDP, p, n, T, Pi, gamma, domain_At, domain_Zt, max_iter=1000, epsilon=1e-3,
        ndim=10, 
    )
    joblib.dump(
        est_IVDM, 'real_data/est_IVDM.pkl'
    )
    joblib.dump(
        est_IVMDP, 'real_data/est_IVMDP.pkl'
    )
    joblib.dump(
        est_MIS, 'real_data/est_MIS.pkl'
    )
    joblib.dump(
        est_DM, 'real_data/est_DM.pkl'
    )
    joblib.dump(
        est_DRL, 'real_data/est_DRL.pkl'
    )
    pass
