import numpy as np
import matplotlib.pyplot as plt
from proj_geometry import *
from utils import *


if __name__ == '__main__':
    #filename = "./data/0042.npz"
    #filename = "./data/0020.npz"    
    filename = "./data/0027.npz"    
    t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
    
#%%    
    
    # (a) IMU Localization via EKF Prediction
    # Time step
    t_size = t.shape[1]
    tau = np.diff(t)
        
    # Random noise for the velocity and the angle    
    rng = np.random.RandomState(seed=1234)
    p_velocity_std = 0.15
    p_omega_std = 0.25
    W_diag = np.array([p_velocity_std,p_velocity_std,p_velocity_std, \
                       p_omega_std,p_omega_std,p_omega_std])
    W = np.diag(W_diag)   
    
    # Initialization of Mean & Covariance matrix
    u_t  = np.concatenate((linear_velocity, rotational_velocity), axis=0)
    mu1_t = np.zeros((4,4,t_size))
    mu1_t[:,:,0] = np.eye(4)
    Sigma1_t = np.zeros((6,6,t_size))
    Sigma1_t[:,:,0] = np.eye(6)    
    
    # Prediction step        
    for i in range(t_size-1):
        ut = -1*tau[0,i]*u_t[:,i]
        noise = (tau[0,i]**2) * W
        mu1_t[:,:,i+1] = SE3(ut) @ mu1_t[:,:,i]
        Sigma1_t[:,:,i+1] = adjSE3(ut) @ Sigma1_t[:,:,i] @ adjSE3(ut).T + noise
        
    T_t_ = np.zeros((4,4,t_size))    
    for i in range(t_size):
        T_t_[:,:,i] = np.linalg.inv(mu1_t[:,:,i])
    T_t = mu1_t          
    
    # Plot the trajectory
    fig, ax = visualize_trajectory_2d(T_t_,filename,True)       
    plt.show()
    plt.waitforbuttonpress()
    plt.close()       
    
#%%
    
    # (b) Landmark Mapping via EKF Update
    # Prepare for known information
    oTi = cam_T_imu
    fsu = K[0,0]
    fsv = K[1,1]    
    cu = K[0,2]
    cv = K[1,2]    
    M = np.zeros((4,4))
    M[0:2,0:3] = K[0:2,0:3]
    M[2: ,0:3] = K[0:2,0:3]    
    M[2,-1] = -1 * K[0,0] * b
    M_dis = np.zeros((3,4))
    M_dis[0:2,0:3] = K[0:2,0:3]
    M_dis[-1,-1] = K[0,0] * b
    lm = features.shape[1]
    D_ = np.zeros((4,3))
    D_[0:3,:] = np.eye(3)    
    D = np.zeros((4*lm,3*lm))    
    for i in range(lm):
        D[(4*i):(4*i+4),(3*i):(3*i+3)] = D_     

    # Initialization of Mean & Covariance matrix
    mu2_t = np.zeros((4,lm,t_size))
    fetur_dis = np.zeros((3,1))
    for i in range(lm):
        for j in range(t_size):
            if ~(features[0,i,j] == -1 and \
                 features[1,i,j] == -1 and \
                 features[2,i,j] == -1 and \
                 features[3,i,j] == -1):
                   uL = features[0,i,j]
                   vL = features[1,i,j]
                   uR = features[2,i,j]
                   vR = features[3,i,j]                  
                   z_z = (fsu*b)/(uL-uR)
                   x_z = z_z * ((uL-cu)/(fsu))
                   y_z = z_z * ((vL-cv)/(fsv))
                   X = np.array([x_z,y_z,z_z,1]).T
                   
                   mu2_tmp = np.linalg.inv(T_t[:,:,j]) @ \
                             np.linalg.inv(oTi)  @ \
                             X
                                                          
                   mu2_t[:,i,0] = mu2_tmp                   
                   break
                           
    sclr = 1
    Sigma2_t = np.zeros((3*lm,3*lm,t_size))
    for i in range(t_size):
        Sigma2_t[:,:,i] = sclr*np.eye(3*lm)
    
    # observed features corresponds to landmarks 
    Nt = np.zeros(t_size)
    corrs = np.zeros((t_size,lm))
    for j in range(t_size):
        for i in range(lm):
            if ~(features[0,i,j] == -1 and \
                 features[1,i,j] == -1 and \
                 features[2,i,j] == -1 and \
                 features[3,i,j] == -1):
                   corrs[j,Nt[j].astype(int)] = i                
                   Nt[j] += 1
    
    # Update step
    for k in range(t_size-1):      
        Ntk = Nt[k].astype(int)
        H_t = np.zeros((4*Ntk,3*lm))
        z = np.zeros((4,Ntk))
        z_hat = np.zeros((4,Ntk))
        V = np.eye(4*Ntk)
        for i in range(Ntk):
            l = corrs[k,i].astype(int)
            dq = deProjection(oTi @ T_t[:,:,k] @ mu2_t[:,l,k])
            H_t[(4*i):(4*i+4),(3*l):(3*l+3)] = M @ dq @ oTi @ \
                                               T_t[:,:,k] @ D_
            z[:,i] = features[:,l,k]
            z_hat[:,i] = M @ Projection(oTi @ T_t[:,:,k] @ mu2_t[:,l,k])
            
        # Random noise for obervation model
        if filename == "./data/0042.npz":
            v_sclr = 500000
        if filename == "./data/0027.npz":
            v_sclr = 500
        if filename == "./data/0020.npz":
            v_sclr = 50      
        
        V = v_sclr * np.eye(4*Ntk)
        K1 = Sigma2_t[:,:,k] @ H_t.T
        K2 = H_t @ Sigma2_t[:,:,k] @ H_t.T
        K_t = K1 @ np.linalg.inv(K2 + V)
        
        mu2_k0 = mu2_t[:,:,k].T.reshape((4*lm,1))        
        z_apox = z - z_hat
        z_apox = z_apox.T.reshape((4*Ntk,1))
        mu2_k1 = mu2_k0 + D @ K_t @ z_apox
        mu2_t[:,:,k+1] = mu2_k1.reshape((lm,4)).T
        Sigma2_t[:,:,k+1] = (np.eye(3*lm)-(K_t @ H_t)) @ Sigma2_t[:,:,k]
    
    # Plot visual mappings of the landmarks
    fig, ax = visualize_trajectory_2d(T_t_,filename,True)             
    s = 1    
    for k in range(t_size):        
        Ntk = Nt[k].astype(int)
        landmark = np.zeros((2,Ntk,t_size))
        for i in range(Ntk):
            l = corrs[k,i].astype(int)            
            landmark[0:2,i,k] = mu2_t[0:2,l,k]
            
        for i in range(landmark.shape[1]):
            s = 1
            ax.scatter(landmark[0,:,k],landmark[1,:,k],s=s,color="green")      
    plt.show()
    plt.waitforbuttonpress()
    plt.close()    

