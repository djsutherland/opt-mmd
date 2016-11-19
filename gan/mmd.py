import tensorflow as tf
import numpy as np

def rbf_mmd2(X, Y, log_sigma=0, biased=True):
    gamma = 1 / (2 * np.exp(2 * log_sigma))
     
    m = X.get_shape()[0]
    n = Y.get_shape()[0]
    
    XX = tf.matmul(X, tf.transpose(X))
    XY = tf.matmul(X, tf.transpose(Y))
    YY = tf.matmul(Y, tf.transpose(Y))

    X_sqnorms = tf.pack([XX[i,i] for i in range(m)])
    Y_sqnorms = tf.pack([YY[i,i] for i in range(n)])

    K_XY = tf.exp(-gamma * (
            -2 * XY + tf.reshape(X_sqnorms, [-1, 1]) + tf.reshape(Y_sqnorms, [1, -1])))
    K_XX = tf.exp(-gamma * (
            -2 * XX + tf.reshape(X_sqnorms, [-1, 1]) + tf.reshape(X_sqnorms, [1, -1])))
    K_YY = tf.exp(-gamma * (
            -2 * YY + tf.reshape(Y_sqnorms, [-1, 1]) + tf.reshape(Y_sqnorms, [1, -1])))

    if biased:
        mmd2 = tf.reduce_mean(K_XX) + tf.reduce_mean(K_YY) - 2 * tf.reduce_mean(K_XY)
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2

def laplace_mmd(X, Y, log_sigma=0, biased=True):
    gamma = 1 / (2 * np.exp(log_sigma))
     
    m = X.get_shape()[0]
    n = Y.get_shape()[0]
    
    XX = tf.matmul(X, tf.transpose(X))
    XY = tf.matmul(X, tf.transpose(Y))
    YY = tf.matmul(Y, tf.transpose(Y))

    X_sqnorms = tf.pack([XX[i,i] for i in range(m)])
    Y_sqnorms = tf.pack([YY[i,i] for i in range(n)])
    X = tf.reshpae(X, [m, 1, -1])
    Y = tf.reshpae(Y, [n, 1, -1])
    K_XY = tf.exp(-gamma * (
            tf.reduce_sum(tf.abs(X - tf.transpose(Y, [1, 0, 2]), 1))))
    K_XX = tf.exp(-gamma * (
            tf.reduce_sum(tf.abs(X - tf.transpose(X, [1, 0, 2]), 1))))
    K_YY = tf.exp(-gamma * (
            tf.reduce_sum(tf.abs(Y - tf.transpose(Y, [1, 0, 2]), 1))))

    if biased:
        mmd2 = tf.reduce_mean(K_XX) + tf.reduce_mean(K_YY) - 2 * tf.reduce_mean(K_XY)
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())

def laplace_mmd2_and_ratio(X, Y, batch_size, log_sigma=None, biased=True):
    K_XY = 0
    K_XX = 0
    K_YY = 0
    m = X.get_shape()[0]
    n = Y.get_shape()[0]

    XX = tf.matmul(X, tf.transpose(X))
    XY = tf.matmul(X, tf.transpose(Y))
    YY = tf.matmul(Y, tf.transpose(Y))
    X_sqnorms = tf.pack([XX[i,i] for i in range(m)])
    Y_sqnorms = tf.pack([YY[i,i] for i in range(n)])
    
    X = tf.reshape(X, [batch_size, 1, -1])
    Y = tf.reshape(Y, [batch_size, 1, -1])
    for sigma_id in range(len(log_sigma)):
      gamma = 1 / (2 * np.exp(log_sigma[sigma_id]))
      K_XY += tf.exp(-gamma * (
            tf.reduce_sum(tf.abs(X - tf.transpose(Y, [1, 0, 2])), 2)))
      K_XX += tf.exp(-gamma * (
            tf.reduce_sum(tf.abs(X - tf.transpose(X, [1, 0, 2])), 2)))
      K_YY += tf.exp(-gamma * (
            tf.reduce_sum(tf.abs(Y - tf.transpose(Y, [1, 0, 2])), 2)))
      

    return _mmd2_and_ratio(K_XX, K_XY, K_YY, batch_size, unit_diagonal=False, biased=biased)

def rbf_mmd2_and_ratio(X, Y, batch_size, log_sigma=None, biased=True):
    K_XY = 0
    K_XX = 0
    K_YY = 0
    m = X.get_shape()[0]
    n = Y.get_shape()[0]

    XX = tf.matmul(X, tf.transpose(X))
    XY = tf.matmul(X, tf.transpose(Y))
    YY = tf.matmul(Y, tf.transpose(Y))
    X_sqnorms = tf.pack([XX[i,i] for i in range(m)])
    Y_sqnorms = tf.pack([YY[i,i] for i in range(n)])
    
    for sigma_id in range(len(log_sigma)):
      gamma = 1 / (2 * np.exp(2 * log_sigma[sigma_id]))
      K_XY += tf.exp(-gamma * (
            -2 * XY + tf.reshape(X_sqnorms, [-1, 1]) + tf.reshape(Y_sqnorms, [1, -1])))
      K_XX += tf.exp(-gamma * (
            -2 * XX + tf.reshape(X_sqnorms, [-1, 1]) + tf.reshape(X_sqnorms, [1, -1])))
      K_YY += tf.exp(-gamma * (
            -2 * YY + tf.reshape(Y_sqnorms, [-1, 1]) + tf.reshape(Y_sqnorms, [1, -1])))


    return _mmd2_and_ratio(K_XX, K_XY, K_YY, batch_size, unit_diagonal=False, biased=biased)

_eps=1e-8

def _mmd2_and_ratio(K_XX, K_XY, K_YY, batch_size, unit_diagonal=False, biased=False,
                    min_var_est=_eps):
    #m = tf.Variable(np.float32(batch_size), trainable=False) 
    m = tf.get_variable("m", (), tf.float32, trainable=False)# Assumes X, Y are same shape
    
    ### Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = tf.pack([K_XX[i,i] for i in range(batch_size)])
        diag_Y = tf.pack([K_YY[i,i] for i in range(batch_size)])

        sum_diag_X = tf.reduce_sum(diag_X)
        sum_diag_Y = tf.reduce_sum(diag_Y)

        sum_diag2_X = tf.reduce_sum(diag_X * diag_X) 
        #diag_X.dot(diag_X)
        sum_diag2_Y = tf.reduce_sum(diag_Y * diag_Y)
        #diag_Y.dot(diag_Y)

    Kt_XX_sums = tf.reduce_sum(K_XX, 1) - diag_X
    Kt_YY_sums = tf.reduce_sum(K_YY, 1) - diag_Y
    K_XY_sums_0 = tf.reduce_sum(K_XY, 0)
    K_XY_sums_1 = tf.reduce_sum(K_XY, 1)

    Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)
    Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)
    K_XY_sum = tf.reduce_sum(K_XY_sums_0)

    # TODO: turn these into dot products?
    # should figure out if that's faster or not on GPU / with theano...
    Kt_XX_2_sum = tf.reduce_sum(tf.square(K_XX)) - sum_diag2_X
    Kt_YY_2_sum = tf.reduce_sum(tf.square(K_YY)) - sum_diag2_Y
    K_XY_2_sum  = tf.reduce_sum(tf.square(K_XY))


    ### Estimators for the various terms involved
    muX_muX = Kt_XX_sum / (m * (m-1))
    muY_muY = Kt_YY_sum / (m * (m-1))
    muX_muY = K_XY_sum / (m * m)

    E_x_muX_sq = (tf.reduce_sum(Kt_XX_sums * Kt_XX_sums) - Kt_XX_2_sum) / (m*(m-1)*(m-2))
    E_y_muY_sq = (tf.reduce_sum(Kt_YY_sums * Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))

    E_x_muY_sq = (tf.reduce_sum(K_XY_sums_1 * K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    E_y_muX_sq = (tf.reduce_sum(K_XY_sums_0 * K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))

    E_x_muX_x_muY = tf.reduce_sum(Kt_XX_sums * K_XY_sums_1) / (m*m*(m-1))
    E_y_muY_y_muX = tf.reduce_sum(Kt_YY_sums * K_XY_sums_0) / (m*m*(m-1))

    E_kxx2 = Kt_XX_2_sum / (m * (m-1))
    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kxy2 = K_XY_2_sum / (m * m)


    ### Combine into the full estimators
    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * muX_muY)
    else:
        mmd2 = muX_muX + muY_muY - 2 * muX_muY

    first_order = 4 * (m-2) / (m * (m-1)) * (
          E_x_muX_sq - tf.square(muX_muX)
        + E_y_muY_sq - tf.square(muY_muY)
        + E_x_muY_sq - tf.square(muX_muY)
        + E_y_muX_sq - tf.square(muX_muY)
        - 2 * E_x_muX_x_muY + 2 * muX_muX * muX_muY
        - 2 * E_y_muY_y_muX + 2 * muY_muY * muX_muY
    )
    second_order = 2 / (m * (m-1)) * (
          E_kxx2 - tf.square(muX_muX)
        + E_kyy2 - tf.square(muY_muY)
        + 2 * E_kxy2 - 2 * muX_muY**2
        - 4 * E_x_muX_x_muY + 4 * muX_muX * muX_muY
        - 4 * E_y_muY_y_muX + 4 * muY_muY * muX_muX
    )
    var_est = first_order + second_order

    ratio = mmd2 / tf.sqrt(tf.maximum(var_est, min_var_est) + 1e-5)
    return mmd2, ratio
