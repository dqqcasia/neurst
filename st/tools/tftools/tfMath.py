import tensorflow as tf


def non_linear(x, non_linear, min_reward=None):
    if non_linear == 'exponential':
        y = tf.exp(x)-1
    elif non_linear == 'prelu':
        y = tf.where(x>0, 2*x, 0.5*x)
    elif non_linear == 'relu':
        y = tf.nn.relu(x)
    elif non_linear == 'm-relu':
        min_reward = tf.ones_like(x) * min_reward
        y = tf.where(x>min_reward, x, min_reward)
    else:
        y = x

    return y


def sum_log(*args):
    """
    Stable log sum exp.
    the input number is in log-scale, so as the return
    """
    # if all(a == LOG_ZERO for a in args):
    #     return LOG_ZERO
    a_max = tf.reduce_max(tf.concat([args], 1), 0)
    lsp = tf.log(tf.reduce_sum([tf.exp(a - a_max) for a in args], 0))
    return a_max + lsp


def testSum_log():
    import numpy as np

    a = tf.constant(np.log([[1.4, 0.2, 1e-11],
                            [1.4, 0.2, 1e-11]]))
    b = tf.constant(np.log([[9.6, 0.02, 1e-11],
                            [9.6, 0.02, 1e-11]]))

    ground_truth = np.array(np.log([11, 0.22, 2e-11]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run(sum_log(a, b))
        print(results)
        print(ground_truth)
        assert np.allclose(results, ground_truth)


def Vander_Monde_matrix(rate, batch_size):
    '''
    [[0, 0, 0, ...]
    [t, t, t, ...]
    [t^2, t^2, t^2, ...]]^T
    ...
    '''
    coefficients = tf.constant([[rate ** t for t in range(100)]])
    matrix = tf.tile(coefficients, [batch_size, 1])

    return matrix


if __name__ == '__main__':
    testSum_log()
