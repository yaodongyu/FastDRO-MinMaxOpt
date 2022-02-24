import numpy as np
from numpy import linalg as LA


def dro_lg_loss(X, y, theta, dro_lambda=0.0, epsilon=0.1, kappa=2.0):
    loss_1 = dro_lambda * epsilon
    loss_2 = (1.0 / X.shape[0]) * np.log(1 + np.exp(-1.0 * np.multiply(y, (X @ theta)))).sum()
    loss_3 = (1.0 / X.shape[0]) * np.maximum(np.multiply(y, (X @ theta)) - dro_lambda * kappa, 0).sum()
    dro_loss = loss_1 + loss_2 + loss_3
    return dro_loss


def proj_theta_lambda(theta, dro_lambda):
    norm_theta = LA.norm(theta, 'fro')
    if norm_theta <= dro_lambda:
        return theta, dro_lambda
    elif norm_theta >= dro_lambda:
        s = (norm_theta + dro_lambda) * 0.5
        theta_proj = (s / norm_theta) * theta
        return theta_proj, s
    else:
        return theta * 0.0, 0.0


def dro_lg_gradient_theta_gda(X, y, theta, nu):
    t0 = np.exp(-1.0 * np.multiply(y, (X@theta)))
    t1 = np.true_divide(-1.0 * np.multiply(y, t0), 1.0 + t0)
    gradient_1 = (1.0/X.shape[0]) * (X.transpose()@t1)
    gradient_2 = (0.5/X.shape[0]) * (X.transpose()@y)
    gradient_3 = (0.5/X.shape[0]) * (X.transpose()@(np.multiply(nu, y)))
    theta_gradient = gradient_1 + gradient_2 + gradient_3
    return theta_gradient


def Operator(X_train, y_train, idx, epsilon, kappa, u, 
             sign_grad_gamma=False):
    lambda_dro = u[0]
    beta = u[1]
    gamma = u[2]

    X_ = X_train[idx, :]
    y_ = y_train[idx]
    gamma_ = gamma[idx]

    batch_size = X_.shape[0] * 1.0
    # calculate gradient of lambda: epsilon - (kappa / 2) * (1 + (1/batch_size) * sum_{i \in Batch_set} gamma_i)
    F_lambda_dro = epsilon - kappa * 0.5 - (0.5/batch_size) * gamma_.sum() * kappa

    # calculate gradient of beta
    F_beta = dro_lg_gradient_theta_gda(X_, y_, beta, gamma_)

    # calculate gradient of gamma_i: - (1/(2*batch_size)) * (y_i * <x_i, beta> - lambda * kappa)
    F_gamma = np.zeros_like(y_train)
    if sign_grad_gamma:
        F_gamma[idx] = np.sign(- (0.5/batch_size) * (np.multiply(y_, (X_ @ beta)) - lambda_dro * kappa))
    else:
        F_gamma[idx] = (- (0.5/batch_size) * (np.multiply(y_, (X_ @ beta)) - lambda_dro * kappa))

    # calculate operator
    F_union = [F_lambda_dro, F_beta, F_gamma]
    return F_union

def Projection_Lambda_Gamma(u):
    lambda_dro = u[0]
    beta = u[1]
    gamma = u[2]
    beta_proj, lambda_dro_proj = proj_theta_lambda(beta, lambda_dro)
    gamma_proj = np.clip(gamma, -1.0, 1.0)

    u_proj = [lambda_dro_proj, beta_proj, gamma_proj]
    return u_proj
