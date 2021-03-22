#!/usr/bin/python
#coding=utf-8
import tensorflow as tf
import logging


def average_gradients(tower_grads):
    average_grads = []
    for tuples_grads_var in zip(*tower_grads):
        # Note that each tuples_grads_var looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        # variable loop
        grads = []
        for i, (grad, var) in enumerate(tuples_grads_var):
            # device loop
            if grad != None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_grad = tf.expand_dims(grad, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_grad)
            else:
                # pass
                logging.warning('here is a variable: {} in gpu_{} which is independent of the loss'.format(var.name, i))

        # Average over the 'tower' dimension.
        if grad != None:
            grad = tf.concat(values=grads, axis=0)
            grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = tuples_grads_var[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def handle_gradients(grads_and_vars, args):
    grads_after_handle = []

    for g, v in grads_and_vars:
        # collect gradient
        # g = tf.scalar_mul(1 / args.trn_streams, g)

        # add l2-penalty
        if args.lambda_l2 > 0 and ('bias' not in v.name) and (g is not None):
            g = tf.add(g, tf.scalar_mul(tf.convert_to_tensor(args.lambda_l2), v))

        # clip gradient value
        if args.grad_clip_value > 0 and "lstm" in v.name:
            g = tf.clip_by_value(g, -args.grad_clip_value, args.grad_clip_value)

        # clip gradient norm
        if args.grad_clip_norm > 0 and 'lstm' in v.name:
            g = tf.clip_by_norm(g, args.grad_clip_norm)

        grads_after_handle.append((g, v))

    # clip global gradient norm
    if args.grad_clip_global_norm > 0:
        list_grads, list_vriables =zip(*grads_after_handle)
        list_global_norm_clipped_grads, _ =tf.clip_by_global_norm(list_grads, args.grad_clip_global_norm)
        grads_after_handle = zip(list_global_norm_clipped_grads, list_vriables)

    return grads_after_handle
