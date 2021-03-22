import tensorflow as tf
import logging
import numpy as np


def size_variables():
    total_size = 0
    all_weights = {v.name: v for v in tf.trainable_variables()}
    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        v_size = int(np.prod(np.array(v.shape.as_list())))
        logging.info("Weight    %s\tshape    %s\tsize    %d" % (v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size))
        total_size += v_size
    logging.info("Total trainable variables size: %d" % total_size)


def smoothing_cross_entropy(logits, labels, vocab_size, confidence):
    """Cross entropy with label smoothing to limit over-confidence."""
    with tf.name_scope("smoothing_cross_entropy"):
        # Low confidence is given to all non-true labels, uniformly.
        low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
        # Normalizing constant is the best cross-entropy value with soft targets.
        # We subtract it just for readability, makes no difference on learning.
        normalizing = -(confidence * tf.log(confidence) + tf.to_float(
            vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
        # Soft targets.
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)
        try:
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)
        except:
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets)
    return xentropy - normalizing


def smoothing_distribution(distributions, vocab_size, confidence):
    share = 1-confidence
    num_targets = tf.reduce_sum(tf.to_int32(distributions>0.0), -1)
    num_zeros = vocab_size - num_targets
    reduce = tf.tile(tf.expand_dims(share/tf.to_float(num_targets), -1), [1, 1, vocab_size])
    add = tf.tile(tf.expand_dims(share/tf.to_float(num_zeros), -1), [1, 1, vocab_size])
    distribution_smoothed = (distributions-reduce)*tf.to_float(distributions>0) +\
            add*tf.to_float(distributions<1e-6)

    return distribution_smoothed

#============================================================================
#  building model
#============================================================================
def choose_device(op, device, default_device):
    # if op.type.startswith('Variable'):
    #     device = default_device
    if op.type in {'Variable', 'VariableV2', 'VarHandleOp'} or op.type.startswith('Variable'):
        device = default_device
    return device

def l2_penalty(iter_variables):
    l2_penalty = 0
    for v in iter_variables:
        if 'biase' not in v.name:
            l2_penalty += tf.nn.l2_loss(v)
    return l2_penalty

#============================================================================
#  learning rate
#============================================================================
def lr_decay_with_warmup(global_step, warmup_steps, hidden_units):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(warmup_steps)
    global_step = tf.to_float(global_step)
    return hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def warmup_exponential_decay(global_step, warmup_steps, peak, decay_rate, decay_steps):
    print('warmup_steps', warmup_steps, 'peak', peak, 'decay_rate', decay_rate, 'decay_steps', decay_steps)
    warmup_steps = tf.to_float(warmup_steps)
    global_step = tf.to_float(global_step)
    # return peak * global_step / warmup_steps
    return tf.where(global_step <= warmup_steps,
                    peak * global_step / warmup_steps,
                    peak * decay_rate ** ((global_step - warmup_steps) / decay_steps))


def stepped_down_decay(global_step, learning_rate, decay_rate, decay_steps):
    decay_rate = tf.to_float(decay_rate)
    decay_steps = tf.to_float(decay_steps)
    learning_rate = tf.to_float(learning_rate)
    global_step = tf.to_float(global_step)

    return learning_rate * decay_rate ** (global_step // decay_steps)


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
    """
    if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger '
                     'or equal to warmup_learning_rate.')
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
                          3.1416 *
                          (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps)
                          / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
    if warmup_steps > 0:
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step, tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)

    return tf.where(global_step > total_steps, 0.0, learning_rate, name='learning_rate')


def exponential_decay(global_step, lr_init, decay_rate, decay_steps, lr_final=None):
    lr = tf.train.exponential_decay(lr_init, global_step, decay_steps, decay_rate, staircase=False)
    if lr_final:
        lr = tf.cond(tf.less(lr, lr_final),
                lambda: tf.constant(lr_final),
                lambda: lr)
    return lr


def create_embedding(size_vocab, size_embedding, name='embedding'):
    if type(size_embedding) == int:
        with tf.device("/cpu:0"):
            embed_table = tf.get_variable(name, [size_vocab, size_embedding])
    else:
        embed_table = None

    return embed_table


if __name__ == '__main__':
    global_step = tf.train.get_or_create_global_step()
    op_add = tf.assign_add(global_step, 1)

    lr = stepped_down_decay(global_step,
                            learning_rate=0.002,
                            decay_rate=0.94,
                            decay_steps=3000)

    with tf.train.MonitoredTrainingSession() as sess:
        list_x = []; list_y = []
        for i in range(50000):
            x, y = sess.run([global_step, lr])
            list_x.append(x)
            list_y.append(y)
            sess.run(op_add)

    import matplotlib.pyplot as plt
    plt.plot(list_x, list_y)
    plt.show()
