from tensorflow.python.ops import gen_math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework import ops
import tensorflow as tf


def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session


def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
    """Return a mask tensor representing the first N positions of each row.

    Example:

    ```python
    tf.sequence_mask([1, 3, 2], 5) =
    [[True, False, False, False, False],
     [True, True, True, False, False],
     [True, True, False, False, False]]
    ```

    Args:
    lengths: 1D integer tensor, all its values < maxlen.
    maxlen: scalar integer tensor, maximum length of each row. Default: use
            maximum over lengths.
    dtype: output type of the resulting tensor.
    name: name of the op.
    Returns:
    A 2D mask tensor, as shown in the example above, cast to specified dtype.

    Raises:
    ValueError: if the arguments have invalid rank.
    """
    with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
        lengths = ops.convert_to_tensor(lengths)

    if maxlen is None:
        maxlen = gen_math_ops._max(lengths, [0])
    else:
        maxlen = ops.convert_to_tensor(maxlen)

    row_vector = gen_math_ops._range(constant(0, maxlen.dtype),
                                     maxlen,
                                     constant(1, maxlen.dtype))
    matrix = gen_math_ops.cast(tf.expand_dims(lengths, 1), maxlen.dtype)
    result = row_vector < matrix

    if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
        return result
    else:
        return gen_math_ops.cast(result, dtype)


def range_batch(shape, range_down=True, dtype=tf.int32):
    """
    sess.run(range_batch([2,5]))
    range_down=False:
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]], dtype=int32)
    range_down=True:
        array([[0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1]], dtype=int32)
    """
    if not range_down:
        result = tf.tile([tf.range(shape[1], dtype=dtype)],
                       [shape[0], 1])
    else:
        result = tf.tile(tf.reshape(tf.range(shape[0], dtype=dtype), (-1, 1)),
                   [1, shape[1]])
    return result


# TODO
def label_smoothing(z, cr=0.8):
    # Label smoothing
    table = tf.convert_to_tensor([[cr, 1.-cr]])

    return tf.nn.embedding_lookup(table, z)


def dense_sequence_to_sparse(seq, len_seq):
    '''convert sequence dense representations to sparse representations
    Args:
        seq: the dense seq as a [batch_size x max_length] tensor
        len_seq: the sequence lengths as a [batch_size] vector
    Returns:
        the sparse tensor representation of the seq

    the reverse op:
        tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=None)
        [[1, 0, 0, 0]
         [0, 0, 2, 0]
         [0, 0, 0, 0]]
        indices：[[0, 0], [1, 2]]
        values：[1, 2]
        dense_shape：[3, 4]
        the default value is `0`

        a_dense = tf.sparse_to_dense(
            sparse_indices=a_sparse.indices,
            output_shape=a_sparse.dense_shape,
            sparse_values=a_sparse.values,
            default_value=0)
    '''
    with tf.name_scope('dense_sequence_to_sparse'):
        #get all the non padding seq
        indices = tf.cast(get_indices(len_seq), tf.int64)
        #create the values
        values = tf.gather_nd(seq, indices)
        #the shape
        shape = tf.cast(tf.shape(seq), tf.int64)
        sparse = tf.SparseTensor(indices, values, shape)

    return sparse


def batch_pad(p, length, pad, direct='head'):
    """
    add the length
    Demo:
        p = tf.ones([4, 3], dtype=tf.float32)
        pad = tf.convert_to_tensor(0.0)
        sess.run(right_length_rows(p, 1, pad, direct='head'))
        array([[ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.]], dtype=float32)
    """
    assert type(length) is int
    if direct == 'head':
        res = tf.concat([tf.fill(dims=[tf.shape(p)[0], length], value=pad), p],
                        axis=1)
    elif direct == 'tail':
        res = tf.concat([p, tf.fill(dims=[tf.shape(p)[0], length], value=pad)],
                        axis=1)
    else:
        raise NotImplementedError

    return res


def batch_pad_to(p, length, pad):
    """
    expend the 2d tensor to assigned length
    """
    length_p = tf.shape(p)[1]
    pad_length = tf.reduce_max([length_p, length])-length_p

    pad = tf.cast(tf.fill(dims=[tf.shape(p)[0], pad_length], value=pad), dtype=p.dtype)
    res = tf.concat([p, pad], axis=1)

    return res


def pad_to(p, length, pad=0.0, axis=1):
    """
    expend the arbitrary shape tensor to assigned length along the assigned axis
    demo:
        p = tf.ones([10, 5])
        pad_to(p, 11)
        <tf.Tensor 'concat_2:0' shape=(10, 11) dtype=float32>
    """
    length_p = tf.shape(p)[axis]
    pad_length = tf.reduce_max([length_p, length])-length_p

    shape = p.get_shape()
    pad_shape = [*shape]
    pad_shape[axis] = pad_length
    pad_tensor = tf.ones(pad_shape, dtype=p.dtype) * pad
    res = tf.concat([p, pad_tensor], axis=axis)

    return res


def pad_to_same(list_tensors):
    """
    pad all the tensors to the same length , given the length info.
    """
    list_lens = []
    for tensor in list_tensors:
        list_lens.append(tf.shape(tensor)[1])
    len_max = tf.reduce_max(tf.stack(list_lens, 0))
    list_padded = []
    for tensor in list_tensors:
        list_padded.append(batch_pad_to(tensor, len_max, 0))

    return list_padded


def right_shift_rows(p, shift, pad):
    assert type(shift) is int

    return tf.concat([tf.fill(dims=[tf.shape(p)[0], 1], value=pad), p[:, :-shift]], axis=1)


def left_shift_rows(p, shift, pad):
    assert type(shift) is int

    return tf.concat([p[:, shift:], tf.fill(dims=[tf.shape(p)[0], 1], value=pad)], axis=1)


def sparse_shrink(sparse, pad=0):
    """
    sparsTensor to shrinked dense tensor:
    from:
     [[x 1 x x 3],
      [2 x x 5 4]]
    to:
     [[1 3 0],
      [2 5 4]]
    """
    dense = tf.sparse_tensor_to_dense(sparse, default_value=-1)
    mask = (dense>=0)
    len_seq = tf.reduce_sum(tf.to_int32(mask), -1)
    indices = get_indices(len_seq)
    values = sparse.values
    shape = [sparse.dense_shape[0], tf.to_int64(tf.reduce_max(len_seq))]
    sparse_shrinked = tf.SparseTensor(indices, values, shape)
    seq = tf.sparse_tensor_to_dense(sparse_shrinked, default_value=pad)

    return seq, len_seq, sparse_shrinked


def acoustic_shrink(distribution_acoustic, len_acoustic, dim_output):
    """
    filter out the distribution where blank_id dominants.
    the blank_id default to be dim_output-1.
    incompletely tested
    the len_no_blank will be set one if distribution_acoustic is all blank dominanted

    """
    blank_id = dim_output - 1
    no_blank = tf.to_int32(tf.not_equal(tf.argmax(distribution_acoustic, -1), blank_id))
    mask_acoustic = tf.sequence_mask(len_acoustic, maxlen=tf.shape(distribution_acoustic)[1], dtype=no_blank.dtype)
    no_blank = mask_acoustic*no_blank
    len_no_blank = tf.reduce_sum(no_blank, -1)

    batch_size = tf.shape(no_blank)[0]
    seq_len = tf.shape(no_blank)[1]

    # the repairing, otherwise the length would be 0
    no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        no_blank,
        tf.concat([tf.ones([batch_size, 1], dtype=tf.int32),
                   tf.zeros([batch_size, seq_len-1], dtype=tf.int32)], 1)
    )
    len_no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        len_no_blank,
        tf.ones_like(len_no_blank, dtype=tf.int32)
    )

    batch_size = tf.size(len_no_blank)
    max_len = tf.reduce_max(len_no_blank)
    acoustic_shrinked_init = tf.zeros([1, max_len, dim_output])

    def step(i, acoustic_shrinked):
        shrinked = tf.gather(distribution_acoustic[i], tf.reshape(tf.where(no_blank[i]>0), [-1]))
        shrinked_paded = pad_to(shrinked, max_len, axis=0)
        acoustic_shrinked = tf.concat([acoustic_shrinked,
                                       tf.expand_dims(shrinked_paded, 0)], 0)

        return i+1, acoustic_shrinked

    i, acoustic_shrinked = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, acoustic_shrinked_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, dim_output])]
    )

    acoustic_shrinked = acoustic_shrinked[1:, :, :]

    return acoustic_shrinked, len_no_blank


def acoustic_hidden_shrink(distribution_acoustic, hidden, len_acoustic, blank_id, hidden_size, num_avg=1):
    """
    filter the hidden where blank_id dominants in distribution_acoustic.
    the blank_id default to be dim_output-1.
    incompletely tested
    the len_no_blank will be set one if distribution_acoustic is all blank dominanted
    shrink the hidden instead of distribution_acoustic
    """
    no_blank = tf.to_int32(tf.not_equal(tf.argmax(distribution_acoustic, -1), blank_id))
    mask_acoustic = tf.sequence_mask(len_acoustic, maxlen=tf.shape(distribution_acoustic)[1], dtype=no_blank.dtype)
    no_blank *= mask_acoustic
    len_no_blank = tf.reduce_sum(no_blank, -1)

    batch_size = tf.shape(no_blank)[0]
    seq_len = tf.shape(no_blank)[1]

    # the patch, the length of shrunk hidden is at least 1
    no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        no_blank,
        tf.concat([tf.ones([batch_size, 1], dtype=tf.int32),
                   tf.zeros([batch_size, seq_len-1], dtype=tf.int32)], 1)
    )
    len_no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        len_no_blank,
        tf.ones_like(len_no_blank, dtype=tf.int32)
    )

    max_len = tf.reduce_max(len_no_blank)
    hidden_shrunk_init = tf.zeros([1, max_len, hidden_size])

    # average the hidden of n frames
    if num_avg == 3:
        hidden = (hidden + \
                tf.concat([hidden[:, 1:, :], hidden[:, -1:, :]], 1) + \
                tf.concat([hidden[:, :1, :], hidden[:, :-1, :]], 1)) / num_avg
    elif num_avg == 5:
        hidden = (hidden + \
                tf.concat([hidden[:, 1:, :], hidden[:, -1:, :]], 1) + \
                tf.concat([hidden[:, 2:, :], hidden[:, -2:, :]], 1) + \
                tf.concat([hidden[:, :2, :], hidden[:, :-2, :]], 1) + \
                tf.concat([hidden[:, :1, :], hidden[:, :-1, :]], 1)) / num_avg

    def step(i, hidden_shrunk):
        # loop over the batch
        shrunk = tf.gather(hidden[i], tf.reshape(tf.where(no_blank[i]>0), [-1]))
        shrunk_paded = pad_to(shrunk, max_len, axis=0)
        hidden_shrunk = tf.concat([hidden_shrunk,
                                   tf.expand_dims(shrunk_paded, 0)], 0)

        return i+1, hidden_shrunk

    i, hidden_shrunk = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, hidden_shrunk_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, hidden_size])]
    )
    hidden_shrunk = hidden_shrunk[1:, :, :]

    return hidden_shrunk, len_no_blank


def alignment_shrink(align, blank_id, pad_id=0):
    """
    //treat the alignment as a sparse tensor where the pad is blank.
    get the indices, values and new_shape
    finally, use the `tf.sparse_tensor_to_dense`

    loop along the batch dim
    """
    batch_size = tf.shape(align)[0]
    len_seq = tf.reduce_sum(tf.to_int32(tf.not_equal(align, blank_id)), -1)

    max_len = tf.reduce_max(len_seq)
    noblank_init = tf.zeros([1, max_len], dtype=align.dtype)

    def step(i, noblank):
        noblank_i = tf.reshape(tf.gather(align[i],
                                         tf.where(tf.not_equal(align[i], blank_id))), [-1])
        pad = tf.ones([max_len-tf.shape(noblank_i)[0]], dtype=align.dtype) * pad_id
        noblank_i = tf.concat([noblank_i, pad], -1)
        noblank = tf.concat([noblank, noblank_i[None, :]], 0)

        return i+1, noblank

    _, noblank = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, noblank_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None])]
    )

    return noblank[1:], len_seq


def get_indices(len_seq):
    '''get the indices corresponding to sequences (and not padding)
    Args:
        len_seq: the len_seqs as a N-D tensor
    Returns:
        A [sum(len_seq) x N-1] Tensor containing the indices'''

    with tf.name_scope('get_indices'):

        numdims = len(len_seq.shape)

        #get the maximal length
        max_length = tf.reduce_max(len_seq)

        sizes = tf.shape(len_seq)

        range_tensor = tf.range(max_length)
        for i in range(1, numdims):
            tile_dims = [1]*i + [sizes[i]]
            range_tensor = tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

        indices = tf.where(tf.less(range_tensor,
                                   tf.expand_dims(len_seq, numdims)))

    return indices