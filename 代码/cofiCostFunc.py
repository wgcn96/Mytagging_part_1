import numpy as np
# get the cost function & gradient
def cofiCostFunc(params,Y,R,num_users,num_movies,num_features,lamda):
    # Unfold the U and W matrices
    x_part=params[0:num_movies*num_features]
    X=np.mat(x_part).reshape(num_movies,num_features)
    theta_part=params[num_movies*num_features:]
    Theta=np.mat(theta_part).reshape(num_users,num_features)


    # compute the following values and return
    J=0
    X_grad=np.mat(np.zeros(X.shape))
    Theta_grad=np.mat(np.zeros(Theta.shape))


    temp0=X*Theta.T-Y
    temp1=np.multiply(temp0,temp0) # 矩阵点乘
    temp2=np.multiply(temp1,R)
    J=(1/2)*temp2.sum()
    X_grad=np.multiply(temp0,R)*Theta
    Theta_grad=(np.multiply(temp0,R)).T*X

    # with Regulization
    J=J+(lamda/2)*(np.multiply(Theta,Theta).sum())+(lamda/2)*(np.multiply(X,X).sum())
    X_grad=X_grad+lamda*X
    Theta_grad=Theta_grad+lamda*Theta

    # unfold X_grad & Theta_grad
    grad = unfold(X_grad,Theta_grad)
    return J

def gradient(params,Y,R,num_users,num_movies,num_features,lamda):
    # Unfold the U and W matrices
    x_part=params[0:num_movies*num_features]
    X=np.mat(x_part).reshape(num_movies,num_features)
    theta_part=params[num_movies*num_features:]
    Theta=np.mat(theta_part).reshape(num_users,num_features)


    # compute the following values and return
    J=0
    X_grad=np.mat(np.zeros(X.shape))
    Theta_grad=np.mat(np.zeros(Theta.shape))


    temp0=X*Theta.T-Y
    temp1=np.multiply(temp0,temp0) # 矩阵点乘
    temp2=np.multiply(temp1,R)
    J=(1/2)*temp2.sum()
    X_grad=np.multiply(temp0,R)*Theta
    Theta_grad=(np.multiply(temp0,R)).T*X

    # with Regulization
    J=J+(lamda/2)*(np.multiply(Theta,Theta).sum())+(lamda/2)*(np.multiply(X,X).sum())
    X_grad=X_grad+lamda*X
    Theta_grad=Theta_grad+lamda*Theta

    # unfold X_grad & Theta_grad
    grad = unfold(X_grad,Theta_grad)
    return grad

def unfold(X,Theta):
    u=[]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u.append(X[i,j])
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            u.append(Theta[i,j])
    return u
