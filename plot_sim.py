import numpy as np
import joblib
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# eta_pi_true = 167.27288324516914
# eta_pi_true = -2.9322280977898183
# eta_pi_true = -2.547603823888985
eta_pi_true = -2.653336719492989

dir_path = "real_data"

est_IVDM=joblib.load('{}/est_IVDM.pkl'.format(dir_path))
est_IVMDP=joblib.load('{}/est_IVMDP.pkl'.format(dir_path))
est_DM=joblib.load('{}/est_DM.pkl'.format(dir_path))
est_MIS=joblib.load('{}/est_MIS.pkl'.format(dir_path))
est_DRL=joblib.load('{}/est_DRL.pkl'.format(dir_path))
# est_IVMDP[range(2),:]
# est_DM[range(2),:]
# est_MIS[range(2),:]
# est_DRL[range(2),:]
# est_IVMDP=est_IVMDP[range(2),:]
# est_DM=est_DM[range(2),:]
# est_MIS=est_MIS[range(2),:]
# est_DRL=est_DRL[range(2),:]

res = [
    est_IVDM,
    est_IVMDP,
    est_DM,
    est_MIS,
    est_DRL,
]

MSE_IVDM, MSE_IVMDP, MSE_DM, MSE_MIS, MSE_DRL = [
    np.mean((x-eta_pi_true)**2, 0) for x in res
]
Bias_IVDM, Bias_IVMDP, Bias_DM, Bias_MIS, Bias_DRL = [
    np.mean(x-eta_pi_true, 0) for x in res
]

# MSE_IVMDP=np.sum((est_IVMDP[:,range(6,10)]-eta_pi_true)**2,0)/2
# MSE_DM=np.sum((est_DM[:,range(6,10)]-eta_pi_true)**2,0)/2
# MSE_MIS=np.sum((est_MIS[:,range(6,10)]-eta_pi_true)**2,0)/2
# MSE_DRL=np.sum((est_DRL[:,range(6,10)]-eta_pi_true)**2,0)/2

# Bias_IVMDP=np.sum((est_IVMDP[:,range(6,10)]-eta_pi_true),0)/2
# Bias_DM=np.sum((est_DM[:,range(6,10)]-eta_pi_true),0)/2
# Bias_MIS=np.sum((est_MIS[:,range(6,10)]-eta_pi_true),0)/2
# Bias_DRL=np.sum((est_DRL[:,range(6,10)]-eta_pi_true),0)/2

import matplotlib.pyplot as plt
 
trajectory = [100,200,300,400,500,600,700,800,900,1000]

fig, (ax1, ax2) = plt.subplots(1, 2)

#pyl.xlim(0, 20)
#pyl.ylim(0.25, 2)

ax1.plot(trajectory, np.log(MSE_IVDM/eta_pi_true**2) ,marker='*', mec='r', mfc='r', label='IVDM')
ax1.plot(trajectory, np.log(MSE_IVMDP/eta_pi_true**2) ,marker='*', mec='r', mfc='r', label='IVMDP')
ax1.plot(trajectory, np.log(MSE_DM/eta_pi_true**2) ,marker='*', mec='r', mfc='r', label='DM')
ax1.plot(trajectory, np.log(MSE_MIS/eta_pi_true**2) ,marker='*', mec='r', mfc='r', label='MIS')
ax1.plot(trajectory, np.log(MSE_DRL/eta_pi_true**2) ,marker='*', mec='r', mfc='r', label='DRL')
ax1.set_xlabel("number of trajectories")
ax1.set_title('log Relative MSE')

ax2.plot(trajectory, Bias_IVDM/abs(eta_pi_true),marker='*', mec='r', mfc='r', label='IVDM')
ax2.plot(trajectory, Bias_IVMDP/abs(eta_pi_true),marker='*', mec='r', mfc='r', label='IVMDP')
ax2.plot(trajectory, Bias_DM/abs(eta_pi_true),marker='*', mec='r', mfc='r', label='DM')
ax2.plot(trajectory, Bias_MIS/abs(eta_pi_true),marker='*', mec='r', mfc='r', label='MIS')
ax2.plot(trajectory, Bias_DRL/abs(eta_pi_true),marker='*', mec='r', mfc='r', label='DRL')
ax2.set_xlabel("number of trajectories")
ax2.set_title('Relative Bias')
plt.legend(loc='best')
plt.show()