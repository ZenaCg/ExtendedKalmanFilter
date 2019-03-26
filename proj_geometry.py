import numpy as np

def so3(theta):
    
    theta_skew = np.zeros((3,3))
    theta_skew[0,1] = -1 * theta[2]
    theta_skew[0,2] =  1 * theta[1]
    theta_skew[1,0] =  1 * theta[2]
    theta_skew[1,2] = -1 * theta[0]
    theta_skew[2,0] = -1 * theta[1]
    theta_skew[2,1] =  1 * theta[0]     
    
    so3_theta = theta_skew
    
    return so3_theta

def se3(xi):
    
    rho   = xi[0:3]
    theta = xi[3: ]
    se3_xi = np.zeros((4,4))
    se3_xi[0:3,0:3] = so3(theta)
    se3_xi[0:3,3]   = rho
    
    return se3_xi


def SO3(theta):
    
    ntheta = np.linalg.norm(theta)
    theta2 = theta @ theta.T - (theta.T @ theta) * np.eye(3)
    
    R1 = (np.sin(ntheta)/ntheta) * so3(theta)
    R2 = ((1-np.cos(ntheta))/(ntheta**2)) * theta2
    SO3_theta = np.eye(3) + R1 + R2
    
    return SO3_theta


def JacobianL(theta):
    
    ntheta = np.linalg.norm(theta)    
    theta2 = theta @ theta.T - (theta.T @ theta) * np.eye(3)    
    
    J1 = ((1-np.cos(ntheta))/(ntheta**2)) * so3(theta)
    J2 = ((ntheta-np.sin(ntheta))/(ntheta**3)) * theta2
    JL_theta = np.eye(3) + J1 + J2
    
    return JL_theta


def SE3(xi):
    
    theta = xi[3: ]
    rho   = xi[0:3]
    SE3_xi = np.zeros((4,4)) 
    SE3_xi[0:3,0:3] = SO3(theta)   
    SE3_xi[0:3,-1]  = JacobianL(theta) @ rho.T
    SE3_xi[-1 ,-1]  = 1
    
    return SE3_xi


def adjse3(xi):
    
    theta = xi[3: ]
    rho   = xi[0:3]
    adjse3_xi = np.zeros((6,6))
    adjse3_xi[0:3,0:3] = so3(theta)
    adjse3_xi[3: ,3: ] = so3(theta)    
    adjse3_xi[0:3,3: ] = so3(rho)   
    
    return adjse3_xi


def adjSE3(xi):
    
    theta = xi[3: ]
    rho   = xi[0:3]
    adjSE3_xi = np.zeros((6,6))
    adjSE3_xi[0:3,0:3] = SO3(theta)
    adjSE3_xi[3: ,3: ] = SO3(theta)
    adjSE3_xi[0:3,3: ]  = so3(JacobianL(theta)@rho.T) @ SO3(theta)
    
    return adjSE3_xi


def deProjection(q):
    
    dq = np.eye(4)
    dq[2,2] = 0
    dq[0,2] = -1*q[0]/q[2]
    dq[1,2] = -1*q[1]/q[2]
    dq[3,2] = -1*q[3]/q[2]    
    dq = dq/q[2]
    
    
    return dq


def Projection(q):
    
    return q/q[2]

