import tensorflow as tf
import numpy as np


def pad_to(list_array, max_len, size_pad):
    """
    concate list of same size (size_pad) array pad to a larger array
    """
    pad = np.zeros((max_len-len(list_array), size_pad), np.float32)
    if list_array:
        res = np.concatenate([np.array(list_array), pad], 0)
    else:
        res = pad

    return res


def pad_to_same(list_array):
    """
    concate list of same size (size_pad) array pad to a larger array
    """
    list_sents = []
    if list_array:
        max_len = max(len(array) for array in list_array)

        for array in list_array:
            pad = np.zeros((max_len-len(array)), dtype=np.int32)
            list_sents.append(np.concatenate([array, pad]))

    return np.array(list_sents, dtype=np.int32)


def add_avg_hidden(list_hidden, list_frames):
    if list_frames:
        frames = list_frames
        # frames, weights = list(zip(*list_frames))
        h = np.average(frames, axis=0)
        list_hidden.append(h)
        del list_frames[:]


def add_weighted_avg_hidden(list_hidden, list_frames):
    if list_frames:
        frames, weights = list(zip(*list_frames))
        h = np.average(frames, axis=0, weights=weights)
        list_hidden.append(h)
        del list_frames[:]


def add_middle_hidden(list_hidden, list_frames):
    if list_frames:
        frames = list_frames
        # frames, weights = list(zip(*list_frames))
        h = frames[int(len(list_frames)/2)]
        list_hidden.append(h)
        del list_frames[:]


def add_7_middle_hiddens(list_hidden, list_frames, frames):
    if list_frames:
        null = np.zeros_like(frames[0])
        h = np.concatenate(list(frames) + [null]*(7-len(frames)))
        list_hidden.append(h)
        del list_frames[:]


def add_concate_hidden(list_hidden, list_frames, num=4):
    if list_frames:
        # frames, _ = list(zip(*list_frames))
        frames = list_frames
        null = np.zeros_like(frames[0])
        h = np.concatenate(list(frames[:num]) + [null]*(num-len(frames)))
        list_hidden.append(h)
        del list_frames[:]


def acoustic_hidden_shrink(hidden, distribution_acoustic, alignments, len_acoustic, blank_id, num_frames=1):
    """
    alignments: [b x t]
    distribution_acoustic: [b x t x v]
    hidden: [b x t x h]
    num_post: the number of posterior hidden frame that add to the current one
    """
    list_batch = []
    list_len = []
    # batch loop
    for i, _hidden in enumerate(hidden):
        list_hidden = []
        token_pre = None
        list_frames = []
        # time loop
        for t, h in zip(range(len_acoustic[i]), _hidden):
            if alignments[i][t] == token_pre:
                # a repeated no blank
                # p = distribution_acoustic[i][t][token_pre]
                list_frames.append(h)

            elif alignments[i][t] == blank_id:
                # a blank
                token_pre = None
                # if num_post>0 and list_frames:
                #     list_frames.extend([*_hidden[t+1: min(t+num_post+1, len_acoustic[i]-1)]])
                # add_middle_hidden(list_hidden, list_frames)
                # add_7_middle_hiddens(list_hidden, list_frames, _hidden[max(0,t-3):t+4])
                # add_avg_hidden(list_hidden, list_frames)
                add_concate_hidden(list_hidden, list_frames, num_frames)
                # add_weighted_avg_hidden(list_hidden, list_frames)
            else:
                # a new no blank
                # add_middle_hidden(list_hidden, list_frames)
                # add_7_middle_hiddens(list_hidden, list_frames, _hidden[max(0,t-3):t+4])
                # add_avg_hidden(list_hidden, list_frames)
                add_concate_hidden(list_hidden, list_frames, num_frames)
                # add_weighted_avg_hidden(list_hidden, list_frames)
                token_pre = alignments[i][t]
                # p = distribution_acoustic[i][t][token_pre]
                list_frames = [h]

        # add_middle_hidden(list_hidden, list_frames)
        # add_7_middle_hiddens(list_hidden, list_frames, _hidden[max(0,t-3):t+4])
        # add_avg_hidden(list_hidden, list_frames)
        add_concate_hidden(list_hidden, list_frames, num_frames)
        # add_weighted_avg_hidden(list_hidden, list_frames)

        list_batch.append(list_hidden)
        list_len.append(len(list_hidden))

    list_hidden_padded = []
    for hidden_shrunk in list_batch:
        # the hidden is at least with length of 1
        hidden_padded = pad_to(hidden_shrunk, max(list_len+[1]), num_frames * hidden.shape[-1])
        list_hidden_padded.append(hidden_padded)
    acoustic_shrunk = np.stack(list_hidden_padded, 0)
    # the hidden is at least with length of 1
    list_len = [x if x != 0 else 1 for x in list_len]

    del list_batch, list_hidden_padded

    return acoustic_shrunk, np.array(list_len, np.int32)


def acoustic_hidden_shrink_tf(distribution_acoustic, hidden, len_acoustic, blank_id, frame_expand):
    """
    NOTATION: the gradient will not pass over the input vars
    """
    alignments = tf.argmax(distribution_acoustic, -1)
    hidden_shrunk, len_no_blank = tf.py_func(acoustic_hidden_shrink, [hidden, distribution_acoustic, alignments, len_acoustic, blank_id, frame_expand],
                      (tf.float32, tf.int32))
    hidden_shrunk.set_shape([None, None, frame_expand*hidden.get_shape()[-1]])
    len_no_blank.set_shape([None])

    return hidden_shrunk, len_no_blank


def add_blank(list_num_tokens, list_token_musks, num_repeated, align, i):
    if num_repeated > 0:
        list_num_tokens.append(num_repeated)
        musk = np.zeros_like(align, dtype=np.float32)
        musk[i-num_repeated+1: i+1] = np.ones([num_repeated], dtype=np.float32)
        list_token_musks.append(musk)


def analysis_alignments(alignments, len_acoustic, blank_id):
    """
    t: acoustic length
    u: label length
    input:
        alignments: [b, t]
    output:
        num_repeated_frames: [b, u]
        len_label: [b]
        musk_repeated: [bxu, t]
    """
    lists_num_tokens = []
    list_token_len = []
    list_token_musks = []

    for align, len_time in zip(alignments, len_acoustic):
        list_num_tokens = []
        num_repeated = 0
        token_pre = None
        # the end of align is set to blank
        for i, token in zip(range(len_time), align):
            if token == token_pre:
                # repteted token
                num_repeated +=1
            elif token == blank_id and num_repeated>0:
                # current char ends
                add_blank(list_num_tokens, list_token_musks, num_repeated, align, i-1)
                num_repeated = 0
            elif token == blank_id and num_repeated == 0:
                # naive blank
                pass
            else:
                # new token
                add_blank(list_num_tokens, list_token_musks, num_repeated, align, i-1)
                token_pre = token
                num_repeated = 1

        add_blank(list_num_tokens, list_token_musks, num_repeated, align, i)

        if not list_num_tokens:
            # all the sent is blank
            list_num_tokens.append(1)
            list_token_musks.append(np.ones_like(align, dtype=np.float32))

        lists_num_tokens.append(list_num_tokens)
        list_token_len.append(len(list_num_tokens))

    num_repeated_frames = pad_to_same(lists_num_tokens)
    len_label = np.array(list_token_len, np.int32)
    musk_repeated = pad_to_same(list_token_musks)

    return num_repeated_frames, len_label, musk_repeated


def analysis_alignments_tf(alignments, len_acoustic, blank_id):
    num_repeated_frames, len_label, musk_repeated = tf.py_func(
        analysis_alignments, [alignments, len_acoustic, blank_id],
        (tf.int32, tf.int32, tf.int32))
    num_repeated_frames.set_shape([None, None])
    len_label.set_shape([None])
    musk_repeated.set_shape([None, alignments.get_shape()[-1]])

    return num_repeated_frames, len_label, musk_repeated


def acoustic_hidden_shrink_v2(distribution_acoustic, hidden, len_acoustic, blank_id):
    """
    alignments: [b, t]
    hidden: [b x t x h]
    num_repeated_frames: [b x u]
    len_label: [b]
    musk_repeated: [bxu, t]

    asserts : 1. step_in_batch should be ended as bxu-1

    NOTATION: each sent in the batch at least has length of 1
    """
    alignments = tf.argmax(distribution_acoustic, -1)
    num_repeated_frames, len_label, musk_repeated = analysis_alignments_tf(alignments, len_acoustic, blank_id)
    batch_size = tf.shape(hidden)[0]
    maxlen_sent = tf.shape(num_repeated_frames)[1]
    size_hidden = hidden.get_shape()[-1]
    frames_shrunk_init = tf.zeros([1, size_hidden])
    acoustic_shrunk_init = tf.zeros([1, maxlen_sent, size_hidden])
    def sent(i, step_in_batch, acoustic_shrunk):

        def step(j, i, step_in_batch, frames_shrunk):
            indices = tf.where(musk_repeated[step_in_batch] > 0)
            frame = tf.reduce_mean(tf.gather(hidden[i], indices[:, 0]), 0)
            frames_shrunk = tf.concat([frames_shrunk,
                                       frame[None, :]], 0)

            return j+1, i, step_in_batch+1, frames_shrunk

        _, _, step_in_batch, frames_shrunk = tf.while_loop(
            cond=lambda j, *_: tf.less(j, len_label[i]),
            body=step,
            loop_vars=[0, i, step_in_batch, frames_shrunk_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([None, size_hidden])])
        frames_shrunk = tf.concat([frames_shrunk[1:],
                                   tf.zeros([maxlen_sent-len_label[i], size_hidden])], 0)
        acoustic_shrunk = tf.concat([acoustic_shrunk,
                                     frames_shrunk[None, :]], 0)

        return i+1, step_in_batch, acoustic_shrunk

    _, _, acoustic_shrunk = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=sent,
        loop_vars=[0, 0, acoustic_shrunk_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([]),
                          tf.TensorShape([None, None, size_hidden])])

    acoustic_shrunk = acoustic_shrunk[1:]

    return acoustic_shrunk, len_label


def acoustic_hidden_shrink_v3(distribution_acoustic, hidden, len_acoustic, blank_id, frame_expand):
    """
    alignments: [b, t]
    hidden: [b x t x h]
    num_repeated_frames: [b x u]
    len_label: [b]
    musk_repeated: [bxu, t]

    asserts : 1. step_in_batch should be ended as bxu-1

    NOTATION: each sent in the batch at least has length of 1
    fix the number of frames for a char is frame_expand
    """
    alignments = tf.argmax(distribution_acoustic, -1)
    _, len_label, musk_repeated = analysis_alignments_tf(alignments, len_acoustic, blank_id)
    batch_size = tf.shape(hidden)[0]
    len_time = tf.shape(hidden)[1]
    maxlen_sent = tf.reduce_max(len_label)
    size_hidden = hidden.get_shape()[-1]*frame_expand
    size_hidden_tf = tf.shape(hidden)[-1]*frame_expand
    frames_shrunk_init = tf.zeros([1, size_hidden])
    acoustic_shrunk_init = tf.zeros([1, maxlen_sent, size_hidden])
    def sent(i, step_in_batch, acoustic_shrunk):

        def step(j, i, step_in_batch, frames_shrunk):
            middle_index = tf.to_int32(tf.reduce_mean(tf.where(musk_repeated[step_in_batch]>0)))
            indices = tf.range(tf.reduce_max([middle_index-int((frame_expand-1)/2), 0]),
                               tf.reduce_min([middle_index+int((frame_expand+1)/2), len_time]))
            frame = tf.reshape(tf.gather(hidden[i], indices), [-1])

            pad = tf.zeros([size_hidden_tf-tf.size(frame)])
            frame = tf.concat([frame, pad], 0)
            frames_shrunk = tf.concat([frames_shrunk,
                                       frame[None, :]], 0)

            return j+1, i, step_in_batch+1, frames_shrunk

        _, _, step_in_batch, frames_shrunk = tf.while_loop(
            cond=lambda j, *_: tf.less(j, len_label[i]),
            body=step,
            loop_vars=[0, i, step_in_batch, frames_shrunk_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([None, size_hidden])])
        frames_shrunk = tf.concat([frames_shrunk[1:],
                                   tf.zeros([maxlen_sent-len_label[i], size_hidden])], 0)
        acoustic_shrunk = tf.concat([acoustic_shrunk,
                                     frames_shrunk[None, :]], 0)

        return i+1, step_in_batch, acoustic_shrunk

    _, _, acoustic_shrunk = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=sent,
        loop_vars=[0, 0, acoustic_shrunk_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([]),
                          tf.TensorShape([None, None, size_hidden])])

    acoustic_shrunk = acoustic_shrunk[1:]

    return acoustic_shrunk, len_label


def acoustic_feature_shrink(distribution_acoustic, feature, len_acoustic, blank_id, frame_expand):
    """
    alignments: [b, t]
    x: [b, 4xt, feat_size]
    num_repeated_frames: [b x u]
    len_label: [b]
    musk_repeated: [bxu, t]

    asserts : 1. step_in_batch should be ended as bxu-1

    NOTATION: each sent in the batch at least has length of 1
    fix the number of frames for a char is 13
    """
    alignments = tf.argmax(distribution_acoustic, -1)
    _, len_label, musk_repeated = analysis_alignments_tf(alignments, len_acoustic, blank_id)
    batch_size = tf.shape(feature)[0]
    len_time = tf.shape(feature)[1]
    maxlen_sent = tf.reduce_max(len_label)
    size_feat = feature.get_shape()[-1]*frame_expand
    size_feat_tf = tf.shape(feature)[-1]*frame_expand
    frames_shrunk_init = tf.zeros([1, size_feat])
    acoustic_shrunk_init = tf.zeros([1, maxlen_sent, size_feat])
    def sent(i, step_in_batch, acoustic_shrunk):

        def step(j, i, step_in_batch, frames_shrunk):
            middle_index = 2*tf.to_int32(tf.reduce_mean(tf.where(musk_repeated[step_in_batch]>0)))
            indices = tf.range(tf.reduce_max([middle_index-6, 0]),
                               tf.reduce_min([middle_index+7, len_time]))
            frame = tf.reshape(tf.gather(feature[i], indices), [-1])
            pad = tf.zeros([size_feat_tf-tf.size(frame)])
            frame = tf.concat([frame, pad], 0)
            frames_shrunk = tf.concat([frames_shrunk,
                                       frame[None, :]], 0)

            return j+1, i, step_in_batch+1, frames_shrunk

        _, _, step_in_batch, frames_shrunk = tf.while_loop(
            cond=lambda j, *_: tf.less(j, len_label[i]),
            body=step,
            loop_vars=[0, i, step_in_batch, frames_shrunk_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([None, size_feat])])
        frames_shrunk = tf.concat([frames_shrunk[1:],
                                   tf.zeros([maxlen_sent-len_label[i], size_feat])], 0)
        acoustic_shrunk = tf.concat([acoustic_shrunk,
                                     frames_shrunk[None, :]], 0)

        return i+1, step_in_batch, acoustic_shrunk

    _, _, acoustic_shrunk = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=sent,
        loop_vars=[0, 0, acoustic_shrunk_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([]),
                          tf.TensorShape([None, None, size_feat])])

    acoustic_shrunk = acoustic_shrunk[1:]

    return acoustic_shrunk, len_label


def constrain_repeated(alignments, hidden, len_acoustic, blank_id):
    """
    alignments: [b, t]
    hidden: [b x t x h]
    repeated_frames: [b x u]
    musk_repeated: [bxu, t]

    asserts : 1. step_in_batch should be ended as bxu-1
    """
    num_repeated_frames, len_label, musk_repeated = analysis_alignments_tf(alignments, len_acoustic, blank_id)
    batch_size = tf.shape(hidden)[0]

    def sent(i, step_in_batch, loss):

        def step(j, i, step_in_batch, loss_sent):
            indices = tf.where(musk_repeated[step_in_batch] > 0)
            repeated_frames = tf.gather(hidden[i], indices)
            mean, variance = tf.nn.moments(repeated_frames, 0)
            loss_sent += tf.reduce_sum(variance)

            return j+1, i, step_in_batch+1, loss_sent

        _, _, step_in_batch, loss_sent = tf.while_loop(
            cond=lambda j, *_: tf.less(j, len_label[i]),
            body=step,
            loop_vars=[0, i, step_in_batch, 0.0],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([]),
                              tf.TensorShape([])])
        loss += loss_sent

        return i+1, step_in_batch, loss

    _, _, loss = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=sent,
        loop_vars=[0, 0, 0.0],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([]),
                          tf.TensorShape([])])

    return loss


def repeated_constrain_loss(distribution_acoustic, hidden, len_acoustic, blank_id):
    """
    constrain the repeated hidden representations to be the same
    """
    alignments = tf.argmax(distribution_acoustic, -1) # alignments: [b x t]
    loss = constrain_repeated(
        alignments=alignments,
        hidden=hidden,
        len_acoustic=len_acoustic,
        blank_id=blank_id)

    return loss


def test_acoustic_hidden_shrink():
    distribution_acoustic = np.array(
        [[[0.04,0.01, 0.05, 0.9],
          [0.3, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.05,0.01, 0.9, 0.04]],
         [[0.9, 0.01, 0.05, 0.04],
          [0.1, 0.05, 0.05, 0.9],
          [0.1, 0.05, 0.05, 0.9],
          [0.05,0.01, 0.9, 0.04]]])
    len_acoustic = [2, 4]
    res = acoustic_hidden_shrink(distribution_acoustic, distribution_acoustic, len_acoustic, 3)
    print(res)


def test_shrink_tf():
    distribution_acoustic = tf.constant(
        [[[0.04,0.01, 0.05, 0.9],
          [0.3, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.05,0.01, 0.9, 0.04]],
         [[0.9, 0.01, 0.05, 0.04],
          [0.1, 0.05, 0.05, 0.9],
          [0.1, 0.05, 0.05, 0.9],
          [0.05,0.01, 0.9, 0.04]]])
    len_acoustic = [2, 4]
    res = acoustic_hidden_shrink(distribution_acoustic, distribution_acoustic, len_acoustic, 3)
    print(res)


def test_shrink_v2_tf():

    distribution_acoustic = tf.constant(
        [[[0.04,0.01, 0.05, 0.9],
          [0.3, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.1, 0.8, 0.05, 0.05],
          [0.05,0.01, 0.9, 0.04]],
         [[0.9, 0.01, 0.05, 0.04],
          [0.8, 0.02, 0.05, 0.04],
          [0.1, 0.05, 0.05, 0.9],
          [0.1, 0.05, 0.05, 0.9],
          [0.05,0.01, 0.9, 0.04]]])
    len_acoustic = [5, 5]
    acoustic_shrunk, len_label = acoustic_hidden_shrink_v3(
        distribution_acoustic,
        distribution_acoustic,
        len_acoustic,
        3,
        3)

    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run([acoustic_shrunk, len_label]))


def test_analysis_alignments():
    """
    (array([[2, 1, 3],
            [2, 1, 0]], dtype=int32),
     array([
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=int32))
    """
    alignments = np.array(
    [[9,1,2,9,3,9,4,5,6,9],
     [9,1,2,9,3,9,0,0,0,0]])

    print(analysis_alignments(alignments, [10,6], 9))


def test_constrain_repeated():
    """
    [(None,
        <tf.Variable 'Variable:0' shape=(2, 10) dtype=int32_ref>),
    (<tf.Tensor 'gradients/while/while/strided_slice_2/Enter_grad/b_acc_3:0' shape=(2, 10, 3) dtype=float32>,
        <tf.Variable 'Variable_1:0' shape=(2, 10, 3) dtype=float32_ref>)]
    the gradient can flow the hidden representation rather the alignment, which
    means the model could learn the acoutic representaton but it can only depend
    on the CTC loss to learning the alignment
    """

    # alignments = tf.Variable(
    #     [[9,1,2,9,3,9,4,5,6,9],
    #      [9,1,2,9,3,9,0,0,0,0],
    #      [9,9,9,9,9,9,0,0,0,0]],
    #     dtype=np.int32)
    alignments = tf.Variable(
        [[9,9,9,9,9,9,9,9,9,9],
         [9,9,9,9,9,9,0,0,0,0],
         [9,9,9,9,9,9,0,0,0,0]],
        dtype=np.int32)
    hidden = tf.Variable([
        [[1,2,3],[3,3,3],[3,3,3],[1,2,3],[2,10,20],[1,2,3],[2,20,2],[2,20,2],[2,20,2],[1,2,3]],
        [[1,2,3],[2,2,2],[2,2,2],[1,2,3],[2,2,2],[1,2,3],[2,2,2],[2,2,2],[3,3,2],[1,2,3]],
        [[1,2,3],[2,2,2],[2,2,2],[1,2,3],[2,2,2],[1,2,3],[2,2,2],[2,2,2],[3,3,2],[1,2,3]]],
        dtype=tf.float32)

    loss = constrain_repeated(alignments, hidden, len_acoustic=[10,6,6], blank_id=9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # gradients = optimizer.compute_gradients(loss)
    # print(gradients)
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss))


if __name__ == '__main__':
    # test_acoustic_hidden_shrink()
    # test_analysis_alignments()
    # test_constrain_repeated()
    test_shrink_v2_tf()
