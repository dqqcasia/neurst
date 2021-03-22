import numpy as np
from python_speech_features import mfcc, logfbank
import soundfile

def load_wavfile(wavfile):
    """
    Read a wav file using scipy.io.wavfile
    """
    if wavfile.endswith('.wav'):
        rate, sig = wav.read(wavfile)
    elif wavfile.endswith('.flac'):
        sig, rate = soundfile.read(wavfile, dtype='int16')
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(wavfile))
    # data_name = os.path.splitext(os.path.basename(wavfile))[0]
    return rate, sig


def get_duration(wavfile):
    """
    The duration is equal to the number of frames divided by the framerate (frames per second):
    ```
    import wave
    import contextlib
    fname = '/tmp/test.wav'
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        print(duration)
    ```
    """
    rate, sig = load_wavfile(wavfile)
    return len(sig)/rate

def audio2vector(audio_filename, dim_feature):
    '''
    Turn an audio file into feature representation.
    16k wav, size 283K -> len 903
    '''
    rate, sig = load_wavfile(audio_filename)

    # Get fbank coefficients. numcep is the feature size
    orig_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature).astype(np.float32)
    orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)

    return orig_inputs


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    '''
    Turn an audio file into feature representation.
    '''
    # Load wav files
    rate, sig = load_wavfile(audio_filename)

    # Get mfcc coefficients. numcep is the feature size
    orig_inputs = mfcc(sig, samplerate=rate, numcep=numcep)

    # We only keep every second feature (BiRNN stride = 2) that is: along the length dim with stride 2
    orig_inputs = orig_inputs[::2]

    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current mfcc feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))

    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        # Reminder: array[start:stop:step]
        # slices from indice |start| up to |stop| (not included), every |step|

        # Add empty context data of the correct size to the start and end
        # of the MFCC feature matrix

        # Pick up to numcontext time slices in the past, and complete with empty
        # mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert(len(empty_source_past) + len(data_source_past) == numcontext)

        # Pick up to numcontext time slices in the future, and complete with empty
        # mfcc features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert(len(empty_source_future) + len(data_source_future) == numcontext)

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert(len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)

    # Scale/standardize the inputs
    # This can be done more efficiently in the TensorFlow graph
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    return train_inputs


# np fea opt
def fea_delt1(features):
    feats_padded = np.pad(features, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")

    row, col = np.shape(features)
    l2 = feats_padded[:row]
    l1 = feats_padded[1: row + 1]
    r1 = feats_padded[3: row + 3]
    r2 = feats_padded[4: row + 4]
    delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2

    return delt1


def fea_delt2(features):
    feats_padded = np.pad(features, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")
    feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")

    row, col = np.shape(features)
    l4 = feats_padded[:row]
    l3 = feats_padded[1: row + 1]
    l2 = feats_padded[2: row + 2]
    l1 = feats_padded[3: row + 3]
    c = feats_padded[4: row + 4]
    r1 = feats_padded[5: row + 5]
    r2 = feats_padded[6: row + 6]
    r3 = feats_padded[7: row + 7]
    r4 = feats_padded[8: row + 8]

    delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)

    return delt2


def add_delt(feature):
    fb = []
    fb.append(feature)
    delt1 = fea_delt1(feature)
    # delt1 = np_fea_delt(feature)
    fb.append(delt1)
    # delt2 = np_fea_delt(delt1)
    delt2 = fea_delt2(feature)
    fb.append(delt2)
    fb = np.concatenate(fb, 1)

    return fb


def cmvn_global(feature, mean, var):
    fea = (feature - mean) / var
    return fea


def splice(features, left_num, right_num):
    """
    [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]]
    left_num=0, right_num=2:
        [[1 1 1 2 2 2 3 3 3]
         [2 2 2 3 3 3 4 4 4]
         [3 3 3 4 4 4 5 5 5]
         [4 4 4 5 5 5 6 6 6]
         [5 5 5 6 6 6 7 7 7]
         [6 6 6 7 7 7 0 0 0]
         [7 7 7 0 0 0 0 0 0]]
    """
    dtype = features.dtype
    len_time, dim_raw_feat = features.shape
    stacked_feat = [1]*len_time
    pad_slice = [0.0] * dim_raw_feat
    pad_left = pad_right = []

    for time in range(len_time):
        idx_left = (time-left_num) if time-left_num>0 else 0
        stacked_feat[time] = features[idx_left: time+right_num+1].tolist()
        if left_num - time > 0:
            pad_left = [pad_slice] * (left_num - time)
            stacked_feat[time] = np.concatenate(pad_left+stacked_feat[time], 0)
        elif right_num > (len_time - time - 1):
            pad_right = [pad_slice] * (right_num - len_time + time + 1)
            stacked_feat[time] = np.concatenate(stacked_feat[time]+pad_right, 0)
        else:
            stacked_feat[time] = np.concatenate(stacked_feat[time], 0)

    return np.asarray(stacked_feat, dtype=dtype)


def down_sample(features, rate):

    return features[::rate]


def process_raw_feature(seq_raw_features, args):
    # 1-D, 2-D
    if args.data.add_delta:
        seq_raw_features = add_delt(seq_raw_features)

    # Splice
    fea = splice(
        seq_raw_features,
        left_num=0,
        right_num=args.data.num_context)

    # downsample
    fea = down_sample(
        fea,
        rate=args.data.downsample)

    return fea


if __name__ == '__main__':
    pass
