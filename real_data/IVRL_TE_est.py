# load packages
import joblib
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy.linalg import inv
from sklearn.kernel_approximation import RBFSampler
from numpy.linalg import inv, eigvals

def IVRL_TE_est(MDP, p, n, T, Pi, gamma, domain_At, domain_Zt, max_iter=1000, epsilon=1e-3, ndim=100, l2penalty=1e-3):

    # =============================================================================
    # IV+MDP. Off policy evaluation: estimate the averaged treatment effect in infinite-horizon MDP settings
    # -------------------------------------------------------------------------
    #  parameters:
    #  MDP: dataframe, which contains all observed data formalized 
    #       as (St,Zt,At,Rt) from stage 1 to T. St can be multi-dimensional.
    #  p: the dimension of state variable St
    #  n: the number of trajectories
    #  T: the number of stages
    #  Pi: the target policy we want to evaluate
    #  gamma: discount factor
    #  domain_At: the domain/support of action space
    #  domain_Zt: the domain/support of IV space
    #  max_iter: maximum number of iterations in fitted-Q evaluation
    #  epsilon: error bound in fitted-Q evaluation
    #  
    #  
    # =============================================================================

    # vectorize the data for further use
    if (p==1):
      index_S=np.linspace(0,T-1,T)*4
      index_Splus1=index_S+(p+3)
      Svec=MDP[:,index_S.astype(int)].T.reshape(-1,1)
      Splus1_vec=MDP[:,index_Splus1.astype(int)].T.reshape(-1,1)
    else:
      index_S=np.linspace(0,T-1,T)*(p+3)
      for i in range(1,p):
        index_S=np.concatenate((index_S,(np.linspace(0,T-1,T)*(p+3)+i))).astype(int)
      index_Splus1=index_S+(p+3)
      Svec=MDP[:,index_S].T.reshape(p,-1).T
      Splus1_vec=MDP[:,index_Splus1.astype(int)].T.reshape(p,-1).T

    index_Z=np.linspace(0,T-1,T)*(p+3)+p
    index_A=np.linspace(0,T-1,T)*(p+3)+p+1
    index_R=np.linspace(0,T-1,T)*(p+3)+p+2
    index_SZ=np.concatenate((index_S,index_Z))

    MDP_S=MDP[:,index_S.astype(int)]
    MDP_SZ=MDP[:,index_SZ.astype(int)]
    MDP_A=MDP[:,index_A.astype(int)]

    Zvec=MDP[:,index_Z.astype(int)].T.reshape(-1,1)
    SZvec=np.hstack((Svec,Zvec))
    Avec=MDP[:,index_A.astype(int)].T.reshape(-1,1)
    Rvec=MDP[:,index_R.astype(int)].T.reshape(-1,1)

    nall=n*T

    # estimate models:
    ## (1) estimate Action model p_a(a|s,z) with the first 90% samples
    # A_model = GradientBoostingClassifier(n_estimators = 20, max_depth = 5)
    A_model = LogisticRegression()
    A_model.fit(SZvec[range(int(nall*0.9))], Avec[range(int(nall*0.9))].reshape(1,-1)[0])
    # print out the accuracy of the classifier for p_a(a|s,z) with the last 10% samples
    print("pa model performance (accuracy): ", A_model.score(SZvec[range(int(nall*0.9),nall)], Avec[range(int(nall*0.9),nall)], sample_weight=None),"\n")
    def Pa(St,Zt,n,A_model):
      if ((n==1) and (p==1)):
        return A_model.predict_proba(np.array([St,Zt]).reshape(1, -1))[0]
      elif ((n==1) and (p!=1)):
        return A_model.predict_proba(np.concatenate((St,np.array([Zt]))).reshape(1, -1))[0]
      else:
        return A_model.predict_proba(np.hstack((St,Zt)))


    ## (2) estimate P11 := p_a(At=1|Zt=1,St)
    def P11(St,n,A_model):
      if (n==1):
        return Pa(St,1,n,A_model)[1]
      else:
        return Pa(St,np.ones(n).reshape(-1,1),n,A_model)[:,1]

    ## (3) estimate P10 := p_a(At=1|Zt=0,St)
    def P10(St,n,A_model):
      if (n==1):
        return Pa(St,0,n,A_model)[1]
      else:
        return Pa(St,np.zeros(n).reshape(-1,1),n,A_model)[:,1]

    ## (4) estimate c(Zt|St)
    def c(Zt,St,n,Pi,A_model):
      #ratio = np.clip(ratio, a_min=-100, a_max=100)
      if (n==1):
        c=Zt*((Pi(St)-P10(St,n,A_model))/(P11(St,n,A_model)-P10(St,n,A_model)))+(1-Zt)*((P11(St,n,A_model)-Pi(St))/(P11(St,n,A_model)-P10(St,n,A_model)))
        #return c
        return np.clip(c, a_min=-50, a_max=50)
      else:
        c=Zt.T*((Pi(St)-P10(St,n,A_model))/(P11(St,n,A_model)-P10(St,n,A_model)))+(1-Zt.T)*((P11(St,n,A_model)-Pi(St))/(P11(St,n,A_model)-P10(St,n,A_model)))
        #return c[0]
        return np.clip(c[0], a_min=-50, a_max=50)

    ## (5) estimate pz(Zt|St)
    # Z_model = GradientBoostingClassifier(n_estimators = 20, max_depth = 5)
    Z_model = LogisticRegression()
    Z_model.fit(Svec[range(int(n*T*0.9))], Zvec[range(int(n*T*0.9))].reshape(1,-1)[0])
    print("pz model performance (accuracy): ", Z_model.score(Svec[range(int(n*T*0.9),n*T)], Zvec[range(int(n*T*0.9),n*T)], sample_weight=None),"\n")

    def Pz(Zt,St,n,Z_model):
      if ((n==1) and (p==1)):
        return Z_model.predict_proba(np.array([St]).reshape(1, -1))[0][int(Zt)]
      elif ((n==1) and (p!=1)):
        return Z_model.predict_proba(St.reshape(1, -1))[0][int(Zt)]
      else:
        return Z_model.predict_proba(St)[:,1]*Zt.T[0] + Z_model.predict_proba(St)[:,0]*(1-Zt.T[0])




    ## (6) estimate $\rho(Z_t,S_t)$
    def rho(St,Zt,n,Pi,A_model):
      return c(Zt,St,n,Pi,A_model)/Pz(Zt,St,n,Z_model)

    print("Start estimating omega_pi(St):")
    ## (7) estimate $\omega^{\pi}(s)$: should change if St continuous, and here we only specify linear function class for xi
    class RatioLinearLearner:
        '''
        Input
        --------
        cplearner is an object of rho or PALearner. 
        It gives estimators for conditional probability of behaviour policy: 
        P(action|state) (if input a PALearner), P(mediator|action, state) (if input a PMLearner).
        Examples
        --------
        '''
        def __init__(self, dataset, Pi, time_difference=None, gamma=0.9, ndim=ndim, l2penalty=l2penalty, use_IV=True, rho=rho, A_model=A_model, truncate=100, domain_Zt=np.array([0,1]), domain_At=np.array([0,1])):
            self.use_IV = use_IV

            self.state = np.copy(dataset['state'])
            self.action = np.copy(dataset['action']).reshape(-1, 1)
            self.unique_action = np.unique(dataset['action'])
            if use_IV:
                self.IV = np.copy(dataset['IV']).reshape(-1, 1)
            self.next_state = np.copy(dataset['next_state'])
            self.s0 = np.copy(dataset['s0'])
            if time_difference is None:
                self.time_difference = np.ones(self.action.shape[0])
            else:
                self.time_difference = np.copy(time_difference)

            self.Pi = Pi

            self.gamma = gamma
            self.l2penalty = l2penalty
            self.beta = None
            self.rbf_feature = RBFSampler(random_state=1, n_components=ndim)
            self.rbf_feature.fit(np.vstack((self.state, self.s0)))
            self.truncate = truncate
            pass

        def feature_engineering(self, feature):
            feature_new = self.rbf_feature.transform(feature)
            feature_new = np.hstack([np.repeat(1, feature_new.shape[0]).reshape(-1, 1), feature_new])
            return feature_new


        def fit(self):
            psi = self.feature_engineering(self.state)
            psi_next = self.feature_engineering(self.next_state)
            if self.use_IV:
                ratio = rho(self.state,self.IV,len(self.action),self.Pi,A_model)
            #else:
                  #estimate_pa = self.PALearner.get_pa_prediction(self.state, self.action)
                  #target_pa = self.target_policy_pa(self.policy, self.state, self.action)
                  #pa_ratio = target_pa / estimate_pa
                  #ratio = pa_ratio

            psi_minus_psi_next = self.rbf_difference(psi, psi_next, ratio)
            design_matrix = np.matmul(psi.transpose(), psi_minus_psi_next)
            design_matrix /= self.state.shape[0]
            # print(design_matrix)

            min_eigen_value = np.min(eigvals(design_matrix).real)
            if min_eigen_value >= 1e-5:
                penalty_matrix = np.diagflat(np.repeat(np.abs(min_eigen_value) * self.l2penalty + 1e-5, design_matrix.shape[0]))
            else:
                penalty_matrix = np.diagflat(np.repeat(1e-5, design_matrix.shape[0]))
            # if psi.shape[0] <= psi.shape[1]:
            #     penalty_matrix = np.diagflat(np.repeat(self.l2penalty, design_matrix.shape[0]))
            # else:
            #     penalty_matrix = np.zeros(design_matrix.shape)
            
            penalize_design_matrix = design_matrix + penalty_matrix
            inv_design_matrix = inv(penalize_design_matrix)

            # psi_s0 = self.feature_engineering(self.s0)
            # mean_psi_s0 = (1 - self.gamma) * np.mean(psi_s0, axis=0)
            # print(mean_psi_s0)
            mean_psi_s0 = self.ratio_expectation_s0(np.copy(self.s0))

            beta = np.matmul(inv_design_matrix, mean_psi_s0.reshape(-1, 1))
            self.beta = beta
            pass
        
        def rbf_difference(self, psi, psi_next, ratio):
            # psi_next = self.gamma * (psi_next.transpose() * ratio).transpose()
            psi_next = np.multiply((psi_next.transpose() * ratio).transpose(),
                                  np.power(self.gamma, self.time_difference)[:, np.newaxis])
            psi_minus_psi_next = psi - psi_next
            return psi_minus_psi_next

        def get_ratio_prediction(self, state, normalize=True):
            '''
            Input:
            state: a numpy.array
            Output:
            A 1D numpy array. The probability ratio in certain states.
            '''
            if np.ndim(state) == 0 or np.ndim(state) == 1:
                x_state = np.reshape(state, (1, -1))
            else:
                x_state = np.copy(state)
            psi = self.feature_engineering(x_state)
            ratio = np.matmul(psi, self.beta).flatten()
            ratio_min = 1 / self.truncate
            ratio_max = self.truncate
            ratio = np.clip(ratio, a_min=ratio_min, a_max=ratio_max)
            if state.shape[0] > 1:
                if normalize:
                    ratio /= np.mean(ratio)
            return ratio
        
        def ratio_expectation_s0(self, s0):
            psi_s0 = self.feature_engineering(s0)
            mean_psi_s0 = (1 - self.gamma) * np.mean(psi_s0, axis=0)
            return mean_psi_s0

        def get_r_prediction(self, state, normalize=True):
            return self.get_ratio_prediction(state, normalize)

    state_all=Svec
    IV=Zvec.T[0]
    action=Avec.T[0]
    reward=Rvec.T[0]
    next_state_all=Splus1_vec

    iid_dataset = [state_all, action, IV, reward, next_state_all]

    s0=MDP[:,0:p]
    dataset = {'s0': s0, 'state': state_all, 
            'action': action, 'IV': IV,
            'reward': reward, 
            "next_state": next_state_all}

    
    rlearner = RatioLinearLearner(dataset, Pi, ndim = ndim, use_IV= True, rho=rho, A_model=A_model)
    rlearner.fit()
    omega_all=rlearner.get_ratio_prediction(state_all)

    print("omega_pi(St) estimation finished.\n")


    print("Start estimating Q function and Value function (fitted Q evaluation):")
    ## (8) estimate Q function: we only specify Q as a linear function of (St,Zt,At). Can be generalized later.
    # Q function: a linear function of (St,Zt,At). Can be modified later.
    def Q(beta,n,St,Zt,At):
      if ((n==1) and (p==1)):
        #return np.dot(np.array([St,Zt,At]),beta)
        return np.dot(np.array([1,St,Zt,At]),beta)
      elif ((n==1) and (p!=1)):
        #return np.dot(np.concatenate((St,np.array([Zt]),np.array([At]))),beta)
        return np.dot(np.concatenate((np.array([1]),St,np.array([Zt]),np.array([At]))),beta)
      else:
        #return np.dot(np.hstack((St,Zt,At)),beta)
        return np.dot(np.hstack((np.ones(n).reshape(-1,1),St,Zt,At)),beta)################################3

    ## (9) estimate Value function based on Q function estimation result 
    def V(St, n, Pi, beta, domain_Zt, domain_At, A_model):
      len_Zt=len(domain_Zt)
      len_At=len(domain_At)
      if (n==1):
        V_St=0
        for i in range(len_Zt):
          for j in range(len_At):
            V_St=V_St+c(domain_Zt[i],St,n,Pi,A_model)*Pa(St,domain_Zt[i],n,A_model)[j]*Q(beta,n,St,domain_Zt[i],domain_At[j])
            #V_St=V_St+c(domain_Zt[i],St,n,Pi,A_model)*Pa(St,domain_Zt[i],n,A_model)[j]*Q_model.predict(np.concatenate((St,np.array([domain_Zt[i]]),np.array([domain_At[j]]))))
      else:
        V_St=np.zeros(n)
        for i in range(len_Zt):
          for j in range(len_At):
            V_St=V_St+c(domain_Zt[i]*np.ones(n).reshape(-1,1),St,n,Pi,A_model)*Pa(St,domain_Zt[i]*np.ones(n).reshape(-1,1),n,A_model)[:,j]*Q(beta,n,St,domain_Zt[i]*np.ones(n).reshape(-1,1),domain_At[j]*np.ones(n).reshape(-1,1))
            #V_St=V_St+c(np.ones(n)*domain_Zt[i].reshape(-1,1),St,n,Pi,A_model)*Pa(St,np.ones(n)*domain_Zt[i].reshape(-1,1),n,A_model)[:,j]*Q_model.predict(np.hstack((St,np.ones(n)*domain_Zt[i].reshape(-1,1),np.ones(n)*domain_At[j].reshape(-1,1))))

      return V_St

    # estimate the parameters beta in both Q function and Value function
    nfeatures=p+2
    nall=n*T
    len_Zt=len(domain_Zt)
    len_At=len(domain_At)
    part1_Q=0
    part2_Q=0
    for t in range(T):
      xi_t_Q=np.concatenate((np.ones(n).reshape(-1,1),MDP[:,(p+3)*t:((p+3)*t+p+2)]),axis=1)

      f_tplus1_Q=np.zeros((n,nfeatures+1))
      Stplus1=MDP[:,(p+3)*(t+1):((p+3)*(t+1)+p)]
      for i in range(len_Zt):
        for j in range(len_At):
          xi_tplus1_Q=np.concatenate((np.ones(n).reshape(-1,1),MDP[:,(p+3)*(t+1):((p+3)*(t+1)+p)],domain_Zt[i]*np.ones(n).reshape(-1,1),domain_At[j]*np.ones(n).reshape(-1,1)),axis=1)
          coef_Q=c(domain_Zt[i]*np.ones(n).reshape(-1,1),Stplus1,n,Pi,A_model)*Pa(Stplus1,domain_Zt[i]*np.ones(n).reshape(-1,1),n,A_model)[:,j]
          f_tplus1_Q=f_tplus1_Q+xi_tplus1_Q* coef_Q[:, np.newaxis]

      part1_Q=part1_Q+np.matmul(xi_t_Q.T,(xi_t_Q-gamma*f_tplus1_Q))
      R_t_Q=MDP[:,((p+3)*t+p+2)]
      part2_Q=part2_Q+np.sum(xi_t_Q*R_t_Q[:, np.newaxis],axis=0)

    beta_Q=np.linalg.solve(part1_Q,part2_Q)

    print("Estimated beta_Q is: ", beta_Q)
    print("Q function and Value function estimation finished.\n")


    # calculate the augmentation terms in our DR estimator by function phi_aug
    def phi_aug(MDP,omega_all,rho,Q,V,Pa,c,beta_Q):
      
      phi_result=0
      #phi_result_part1=0
      #phi_result_part2=0

      for t in range(T):
        if (p==1):
          Pa_At=Pa(St=MDP[:,4*t].reshape(-1,1),Zt=MDP[:,4*t+1].reshape(-1,1),n=n, A_model=A_model)
          phi_1=MDP[:,4*t+3]+gamma*V(St=MDP[:,4*t+4].reshape(-1,1), n=n, Pi=Pi, beta=beta_Q, domain_Zt=domain_Zt, domain_At=domain_At, A_model=A_model)
          phi_2=np.zeros(n)
          for i in range(n):
            for a in domain_At:
              phi_2[i]=phi_2[i]+Pa_At[i,a]*Q(beta=beta_Q,n=1,St=MDP[i,4*t],Zt=MDP[i,4*t+1],At=a)

          phi_3=MDP[:,4*t+2]-Pa_At[:,1]
          phi_3_2=np.zeros(n)
          delta_a=(Pa(St=MDP[:,4*t].reshape(-1,1),Zt=np.ones(n).reshape(-1,1),n=n, A_model=A_model)-Pa(St=MDP[:,4*t].reshape(-1,1),Zt=np.zeros(n).reshape(-1,1),n=n, A_model=A_model))[:,1]
          delta_a[np.where(abs(delta_a)<1e-2)]=np.sign(delta_a[np.where(abs(delta_a)<1e-2)])*1e-2
          #delta_a=np.clip(delta_a, a_min=-50, a_max=50)
          for i in range(n):
            for z in domain_Zt:
              for a in domain_At:
                phi_3_2[i]=phi_3_2[i]+(-1)**z/(delta_a[i])*Pa(St=MDP[i,4*t],Zt=z,n=1,A_model=A_model)[a]*Q(beta=beta_Q,n=1,St=MDP[i,4*t],Zt=z,At=a)
          
          phi_3=phi_3*phi_3_2
          #omega_St=omega(MDP[:,(p+3)*t:((p+3)*t+p)],n,p,beta_omega) !!!!!!!!!!!!!!!
          rho_StZt=rho(St=MDP[:,4*t].reshape(-1,1),Zt=MDP[:,4*t+1].reshape(-1,1),n=n,Pi=Pi,A_model=A_model)
        else:
          Pa_At=Pa(St=MDP[:,(p+3)*t:((p+3)*t+p)],Zt=MDP[:,(p+3)*t+p].reshape(-1,1),n=n, A_model=A_model)
          phi_1=MDP[:,(p+3)*t+p+2]+gamma*V(St=MDP[:,(p+3)*(t+1):((p+3)*(t+1)+p)], n=n, Pi=Pi, beta=beta_Q, domain_Zt=domain_Zt, domain_At=domain_At, A_model=A_model)
          phi_2=np.zeros(n)
          for i in range(n):
            for a in domain_At:
              phi_2[i]=phi_2[i]+Pa_At[i,a]*Q(beta=beta_Q,n=1,St=MDP[i,(p+3)*t:((p+3)*t+p)],Zt=MDP[i,(p+3)*t+p],At=a)

          phi_3=MDP[:,(p+3)*t+p+1]-Pa_At[:,1]
          phi_3_2=np.zeros(n)
          delta_a=(Pa(St=MDP[:,(p+3)*t:((p+3)*t+p)],Zt=np.ones(n).reshape(-1,1),n=n, A_model=A_model)-Pa(St=MDP[:,(p+3)*t:((p+3)*t+p)],Zt=np.zeros(n).reshape(-1,1),n=n, A_model=A_model))[:,1]
          delta_a[np.where(abs(delta_a)<1e-2)]=np.sign(delta_a[np.where(abs(delta_a)<1e-2)])*1e-2
          for i in range(n):
            for z in domain_Zt:
              for a in domain_At:
                phi_3_2[i]=phi_3_2[i]+(-1)**z/(delta_a[i])*Pa(St=MDP[i,(p+3)*t:((p+3)*t+p)],Zt=z,n=1,A_model=A_model)[a]*Q(beta=beta_Q,n=1,St=MDP[i,(p+3)*t:((p+3)*t+p)],Zt=z,At=a)
          
          phi_3=phi_3*phi_3_2
          rho_StZt=rho(St=MDP[:,(p+3)*t:((p+3)*t+p)],Zt=MDP[:,(p+3)*t+p].reshape(-1,1),n=n,Pi=Pi,A_model=A_model)
        
        
        phi_result=phi_result+np.sum(1/(1-gamma)*omega_all[t*n:(t+1)*n]*rho_StZt*(phi_1-phi_2+phi_3))

        #phi_result_part1=phi_result_part1+np.sum(1/(1-gamma)*omega_all[t*n:(t+1)*n]*rho_StZt*(phi_1-phi_2))
        #phi_result_part2=phi_result_part2+np.sum(1/(1-gamma)*omega_all[t*n:(t+1)*n]*rho_StZt*phi_3)

      phi_result=phi_result/n/(T)
      #phi_result_part1=phi_result_part1/n/(T)
      #phi_result_part2=phi_result_part2/n/(T)

      return phi_result


    print("Start calculating DM estimator:")
    # calculate the direct estimator
    eta_pi_DM=0
    for i in range(n):
      for z in range(2):
        for a in range(2):
          eta_pi_DM=eta_pi_DM+c(Zt=z,St=MDP[i,0:p],n=1,Pi=Pi,A_model=A_model)*Pa(Zt=z,St=MDP[i,0:p],n=1,A_model=A_model)[a]*Q(beta=beta_Q,n=1,St=MDP[i,0:p],Zt=z,At=a)
    eta_pi_DM=eta_pi_DM/n
    print("DM estimator: eta_pi_DM=", eta_pi_DM)
    print("DM estimator calculation finished.\n")

    print("Start calculating augmentation terms:")
    # calculate the augmentation terms by function phi_aug
    eta_pi_aug=phi_aug(MDP,omega_all,rho,Q,V,Pa,c,beta_Q)
    print("Augmentation term: eta_pi_aug=", eta_pi_aug)
    print("Augmentation terms calculation finished.\n")

    # the final estimator is the summation of DM estimator and augmentation terms, given by eta_pi_DR
    eta_pi_DR=eta_pi_DM+eta_pi_aug
    print("DRL estimator: eta_pi_DR=", eta_pi_DR)

    return eta_pi_DM, eta_pi_DR


