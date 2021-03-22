import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.contrib.rnn import GRUCell, LayerNormBasicLSTMCell, DropoutWrapper, \
    ResidualWrapper, MultiRNNCell, OutputProjectionWrapper
allow_defun = True


def shrink_layer(encoded, len_encoded, logits, dim_hidden):

    batch_size = tf.shape(encoded)[0]
    blank_id = tf.cast(tf.shape(logits)[-1] - 1, tf.int64)
    frames_mark = tf.not_equal(tf.argmax(logits, -1), blank_id)
    prev = tf.concat([tf.ones([batch_size, 1], tf.int64) * blank_id, tf.argmax(logits, -1)[:, :-1]], 1)
    flag_norepeat = tf.not_equal(prev, tf.argmax(logits, -1))
    flag = tf.logical_and(flag_norepeat, frames_mark)
    flag = tf.logical_and(flag, tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.bool))
    len_labels = tf.reduce_sum(tf.cast(flag, tf.int32), -1)
    max_label_len = tf.reduce_max([tf.reduce_max(len_labels), 1])
    hidden_output = tf.zeros([0, max_label_len, dim_hidden], tf.float32)

    def sent(b, hidden_output):
        hidden = tf.gather(encoded[b, :, :], tf.where(flag[b, :])[:, 0])
        pad = tf.zeros([tf.reduce_max([max_label_len - len_labels[b], 0]), dim_hidden])
        hidden_padded = tf.concat([hidden, pad], 0)[:max_label_len, :]
        hidden_output = tf.concat([hidden_output, hidden_padded[None, :]], 0)

        return b+1, hidden_output

    _, hidden_output = tf.while_loop(
    cond=lambda b, *_: tf.less(b, batch_size),
    body=sent,
    loop_vars=[0, hidden_output],
    shape_invariants=[tf.TensorShape([]),
                      tf.TensorShape([None, None, dim_hidden])])

    len_decoded = len_labels

    return hidden_output, len_decoded


def shrink_layer_wdecode(encoded, len_encoded, logits, dim_hidden):

    batch_size = tf.shape(encoded)[0]
    blank_id = tf.cast(tf.shape(logits)[-1] - 1, tf.int64)
    frames_mark = tf.not_equal(tf.argmax(logits, -1), blank_id)
    prev = tf.concat([tf.ones([batch_size, 1], tf.int64) * blank_id, tf.argmax(logits, -1)[:, :-1]], 1)
    flag_norepeat = tf.not_equal(prev, tf.argmax(logits, -1))
    flag = tf.logical_and(flag_norepeat, frames_mark)
    flag = tf.logical_and(flag, tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.bool))
    len_labels = tf.reduce_sum(tf.cast(flag, tf.int32), -1)
    max_label_len = tf.reduce_max([tf.reduce_max(len_labels), 1])
    hidden_output = tf.zeros([0, max_label_len, dim_hidden], tf.float32)
    label_decoded = tf.argmax(logits, -1)
    label_output = tf.zeros([0, max_label_len], tf.int32)

    def sent(b, hidden_output, label_output):
        hidden = tf.gather(encoded[b, :, :], tf.where(flag[b, :])[:, 0])
        pad = tf.zeros([tf.reduce_max([max_label_len - len_labels[b], 0]), dim_hidden])
        hidden_padded = tf.concat([hidden, pad], 0)[:max_label_len, :]
        hidden_output = tf.concat([hidden_output, hidden_padded[None, :]], 0)

        label = tf.gather(label_decoded[b, :, :], tf.where(flag[b, :])[:, 0])
        pad_label = tf.zeros([tf.reduce_max([max_label_len - len_labels[b], 0]), 1])
        label_padded = tf.concat([label, pad_label], 0)


        return b+1, hidden_output, label_output

    _, hidden_output = tf.while_loop(
    cond=lambda b, *_: tf.less(b, batch_size),
    body=sent,
    loop_vars=[0, hidden_output],
    shape_invariants=[tf.TensorShape([]),
                      tf.TensorShape([None, None, dim_hidden]),
                      tf.TensorShape([None, None])])

    len_decoded = len_labels

    return hidden_output, len_decoded


def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
                w_initializer=None, norm_type="batch"):
    with tf.variable_scope(name):
        net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
                           kernel_initializer=w_initializer, name="conv")
        if norm_type == "batch":
            net = tf.layers.batch_normalization(net, name="bn")
        elif norm_type == "layer":
            # net = layer_norm(net)
            net = tf.contrib.layers.layer_norm(net)
        else:
            net = net
        output = tf.nn.relu(net) if use_relu else net

    return output


def block(x, num_filters, i, kernel=(3,9)):
    input = x
    x = normal_conv(
        inputs=x,
        filter_num=num_filters,
        kernel=kernel,
        stride=(1,1),
        padding='SAME',
        use_relu=True,
        name="res_"+str(i),
        norm_type='layer')

    return x+input


def single_cell(num_units, is_train, cell_type, dropout=0.0, forget_bias=0.0, dim_project=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if is_train else 0.0

    # Cell Type
    if cell_type == "lstm":
        single_cell = tf.contrib.rnn.LSTMCell(
            num_units,
            use_peepholes=True,
            num_proj=dim_project,
            cell_clip=50.0,
            forget_bias=1.0)
    elif cell_type == "cudnn_lstm":
        single_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
    elif cell_type == "gru":
        single_cell = GRUCell(num_units)
    elif cell_type == "LSTMBlockCell":
        single_cell = tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=forget_bias)
    elif cell_type == "layer_norm_lstm":
        single_cell = LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    else:
        raise ValueError("Unknown unit type %s!" % cell_type)

    if dim_project:
        single_cell = OutputProjectionWrapper(
            cell=single_cell,
            output_size=dim_project)

    if dropout > 0.0:
        single_cell = DropoutWrapper(cell=single_cell,
                                     input_keep_prob=(1.0 - dropout))

    return single_cell


def build_cell(num_units, num_layers, is_train, cell_type,
               dropout=0.0, forget_bias=0.0, use_residual=False, dim_project=None):
    with tf.name_scope(cell_type):
        list_cell = [single_cell(
            num_units=num_units,
            is_train=is_train,
            cell_type=cell_type,
            dropout=dropout,
            forget_bias=forget_bias,
            dim_project=dim_project) for _ in range(num_layers)]
    # Residual
    if use_residual:
        for c in range(1, len(list_cell)):
            list_cell[c] = ResidualWrapper(list_cell[c])

    return MultiRNNCell(list_cell) if num_layers > 1 else list_cell[0]


def _get_lstm_cell(num_cell_units, is_train, rnn_mode='BLOCK'):
    if rnn_mode == 'BASIC':
        return tf.contrib.rnn.BasicLSTMCell(
            num_cell_units, forget_bias=0.0, state_is_tuple=True,
            reuse=not is_train)
    if rnn_mode == 'BLOCK':
        return tf.contrib.rnn.LSTMBlockCell(
            num_cell_units, forget_bias=0.0)
    if rnn_mode == 'CUDNN':
        return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_cell_units)
    raise ValueError("rnn_mode %s not supported" % rnn_mode)


def make_cell(num_cell_units, is_train, rnn_mode, keep_prob):
    cell = _get_lstm_cell(num_cell_units, is_train, rnn_mode)
    if is_train and keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=keep_prob)
    return cell


def make_multi_cell(num_cell_units, is_train, keep_prob, num_layers, rnn_mode='BLOCK'):
    list_cells = [make_cell(num_cell_units, is_train, rnn_mode, keep_prob) for _ in range(num_layers)]

    multi_cell = tf.contrib.rnn.MultiRNNCell(list_cells, state_is_tuple=True)

    return multi_cell


def cell_forward(cell, inputs, index_layer=0, initial_state=None):
    # the variable created in `tf.nn.dynamic_rnn`, not in cell
    with tf.variable_scope("lstm"):
        lstm_output, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
             initial_state=initial_state,
             scope='cell_'+str(index_layer),
             dtype=tf.float32)
    return lstm_output, state


def conv_lstm(x, kernel_size, filters, padding="SAME", dilation_rate=(1, 1), name='conv_lstm'):
    """Convolutional LSTM in 1 dimension."""
    with tf.variable_scope(name):

        gates = tf.contrib.layers.conv2d(
            x,
            4 * filters,
            (3, 3),
            padding=padding)
        g = tf.split(tf.contrib.layers.layer_norm(gates, 4 * filters), 4, axis=3)
        new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
        hidden_output = tf.sigmoid(g[2]) * tf.tanh(new_cell)

    return hidden_output


def layer_norm_compute_python(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias


@function.Defun(compiled=True)
def layer_norm_compute_grad(x, epsilon, scale, bias, dy):
  y = layer_norm_compute_python(x, epsilon, scale, bias)
  dx = tf.gradients(ys=[y], xs=[x, epsilon, scale, bias], grad_ys=[dy])
  return dx


@function.Defun(
    compiled=True,
    separate_compiled_gradients=True,
    grad_func=layer_norm_compute_grad)
def layer_norm_compute(x, epsilon, scale, bias):
  return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = x.get_shape()[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
    if allow_defun:
      result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
      result.set_shape(x.get_shape())
    else:
      result = layer_norm_compute_python(x, epsilon, scale, bias)
    return result


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4.")
  inputs.set_shape([static_shape[0], None, None, static_shape[3]])
  # Add support for left padding.
  if "padding" in kwargs and kwargs["padding"] == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    kwargs["padding"] = "VALID"
  force2d = False  # Special argument we use to force 2d kernels (see below).
  if "force2d" in kwargs:
    force2d = kwargs["force2d"]

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    if "name" in kwargs:
      original_name = kwargs["name"]
      name = kwargs.pop("name") + "_" + name_suffix
    else:
      original_name = None
      name = "conv_" + name_suffix
    original_force2d = None
    if "force2d" in kwargs:
      original_force2d = kwargs.pop("force2d")
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result


def conv(inputs, filters, kernel_size, **kwargs):
  return conv_internal(tf.layers.conv2d, inputs, filters, kernel_size, **kwargs)


def blstm(hidden_output, len_feas, num_hidden, name):
    num_hidden /= 2

    with tf.variable_scope(name):
        f_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)
        b_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)

        x, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=f_cell,
            cell_bw=b_cell,
            inputs=hidden_output,
            dtype=tf.float32,
            time_major=False,
            sequence_length=len_feas)
        x = tf.concat(x, 2)

    return x


def dense_without_vars(inputs,
                       units,
                       activation=tf.identity,
                       use_bias=True,
                       kernel=None,
                       reuse=None,
                       name=None):
    argcount = activation.__code__.co_argcount
    if activation.__defaults__:
        argcount -= len(activation.__defaults__)
    assert argcount in (0, 1, 2)
    with tf.variable_scope(name, "dense", reuse=reuse):
        if argcount <= 1:
            input_size = inputs.get_shape().as_list()[-1]
            inputs_shape = tf.unstack(tf.shape(inputs))
            inputs = tf.reshape(inputs, [-1, input_size])
            if kernel is not None:
                assert kernel.get_shape().as_list()[0] == units
                w = kernel
            else:
                with tf.variable_scope(tf.get_variable_scope()):
                    w = tf.get_variable("kernel", [units, input_size])
            outputs = tf.matmul(inputs, w, transpose_b=True)
            if use_bias:
                b = tf.get_variable("bias", [units], initializer=tf.zeros_initializer)
                outputs += b
            outputs = activation(outputs)

            return tf.reshape(outputs, inputs_shape[:-1] + [units])
        else:
            arg1 = dense_without_vars(inputs, units, tf.identity, use_bias, name='arg1')
            arg2 = dense_without_vars(inputs, units, tf.identity, use_bias, name='arg2')

            return activation(arg1, arg2)
