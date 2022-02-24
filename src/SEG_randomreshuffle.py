import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import argparse
import random
from utils_dro_logistic_regression import *


def dro_solver(X_train, y_train, 
               kappa=1.0, 
               epsilon=0.1, 
               eta=0.001, 
               batch_size=64, 
               EPOCH=100):
    # Load data and specify hyper parameter
    opt_value = 0.0
    N_EPOCH = EPOCH
    AVG_N = 0.5 * N_EPOCH

    ###### Process data ######
    N, d = X_train.shape

    # Print training data information
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('y values: ', np.unique(y_train))
    print('DRO parameter: kappa={}, epsilon={}'.format(kappa, epsilon))
    print('====================================')

    ###### Set step size ######
    eta_init = eta * 1.0
    shuffle = True

    iteration_total = 1
    avg_iteration_total = 1

    ###### Initialization ######
    dro_lambda_init = 1.0
    beta_init = (np.random.randn(d, 1)) * 0.0
    gamma_init = np.sign(np.multiply(y_train, (X_train @ beta_init)) - dro_lambda_init * kappa)

    dro_losses = []
    dro_losses_bar = []
    plot_n = []
    lambda_list = []

    u = [dro_lambda_init * 1.0, beta_init.copy(), gamma_init.copy()]
    u_init = [dro_lambda_init * 1.0, beta_init.copy(), gamma_init.copy()]
    u_avg = [dro_lambda_init * 1.0, beta_init.copy(), gamma_init.copy()]
    u_bar = [dro_lambda_init * 1.0, beta_init.copy(), gamma_init.copy()]

    plot_n.append(0)
    dro_losses.append(dro_lg_loss(X_train, y_train, u[1], u[0], epsilon, kappa) - opt_value)
    dro_losses_bar.append(dro_lg_loss(X_train, y_train, u_avg[1], u_avg[0], epsilon, kappa) - opt_value)

    # Run algorithm
    for i in range(N_EPOCH):
        number_list = list(range(X_train.shape[0]))
        if shuffle:
            random.shuffle(number_list)

        for idx_ in range(int(X_train.shape[0] / batch_size)):
            # sample
            idx = np.array(number_list[idx_ * batch_size: (idx_ + 1) * batch_size])

            F_idx_u = Operator(X_train, y_train, idx, epsilon, kappa, u)

            # calculate u_bar_{t+1}
            u_bar[0] = u[0] - eta * F_idx_u[0]
            u_bar[1] = u[1] - eta * F_idx_u[1]
            u_bar[2] = u[2] - eta * F_idx_u[2]

            u_bar = Projection_Lambda_Gamma(u_bar)

            F_idx_u_bar = Operator(X_train, y_train, idx, epsilon, kappa, u_bar)

            # calculate u_{t+1}
            u[0] = u[0] - eta * F_idx_u_bar[0]
            u[1] = u[1] - eta * F_idx_u_bar[1]
            u[2] = u[2] - eta * F_idx_u_bar[2]

            u = Projection_Lambda_Gamma(u)

            iteration_total += 1.0

            # averaging
            if i < AVG_N:
                u_avg[0] = u[0]
                u_avg[1] = u[1]
                u_avg[2] = u[2]
            else:
                avg_iteration_total += 1.0
                u_avg[0] = u_avg[0] * (avg_iteration_total + 1.0) / (avg_iteration_total + 2.0) + u[0] * 1.0 / (
                            avg_iteration_total + 2.0)
                u_avg[1] = u_avg[1] * (avg_iteration_total + 1.0) / (avg_iteration_total + 2.0) + u[1] * 1.0 / (
                            avg_iteration_total + 2.0)
                u_avg[2] = u_avg[2] * (avg_iteration_total + 1.0) / (avg_iteration_total + 2.0) + u[2] * 1.0 / (
                            avg_iteration_total + 2.0)

        plot_n.append(i)
        dro_losses.append(dro_lg_loss(X_train, y_train, u[1], u[0], epsilon, kappa) - opt_value)
        dro_losses_bar.append(dro_lg_loss(X_train, y_train, u_avg[1], u_avg[0], epsilon, kappa) - opt_value)
    return u_avg[1], plot_n, dro_losses, dro_losses_bar

