import numpy as np
from features_extraction import FeatFFT
from dataset import load_dataset


def chunking_FFT(features,target,time_window=1,freq=256):
    """
    This function takes as an input the whole dataset of raw data, then chunks i
    t into parts and extracts frequency features, the output is an array with fe
    atures and target

    Parameters
    ----------
    features : numpy array
        array containing features of data, in this case probes from each channel

    target : numpy array
        array containing classes, i.e. which stimuli has been presented

    time_window : integer
        for what time window data is supposed to be chunked

    frequency : integer
        frequency of sampling the data

    """
    classes = np.unique(target)
    C = features.shape[1]
    F = freq
    S = time_window
    X = np.zeros(3) #placeholder-array for results
    t = np.zeros(1)

    for n in classes:
        y = features[np.where(target == n)]
        n_parts = int(y.shape[0]/(F*S))
        for z in range(n_parts):
            part = y[(F*z):(F*(S+z))]
            tmp = FeatFFT().fit_transform(part)
            X = np.vstack((X,tmp))
            t = np.vstack((t,n))
    result = (X[1:],t[1:])

    return result

def load_chunked_dataset():
    """
    This function loads dataset as load_dataset function, then chunks it and ret
    urns it.
    """
    X,y = load_dataset()
    features,target = chunking_FFT(X,y)
    return features,target

def chunking(features,target,time_window=1,freq=256):
    """
    This function takes as an input the whole dataset of raw data, then chunks i
    t into parts and extracts frequency features, the output is an array with fe
    atures and target

    Parameters
    ----------
    features : numpy array
        array containing features of data, in this case probes from each channel

    target : numpy array
        array containing classes, i.e. which stimuli has been presented

    time_window : integer
        for what time window data is supposed to be chunked

    frequency : integer
        frequency of sampling the data

    """
    classes = np.unique(target)
    C = features.shape[1]
    F = freq
    S = time_window
    X = np.zeros(C) #placeholder-array for results
    t = np.zeros(1)

    for n in classes:
        y = features[np.where(target == n)]
        n_parts = int(y.shape[0]/(F*S))
        for z in range(n_parts):
            part = y[(F*z):(F*(S+z))]
            X = np.vstack((X,part))
            t = np.vstack((t,n))

    X = X[1:]
    t = t[1:]
    X = X.reshape(t.shape[0],(F*S),C)
    result = (X,t)

    return result


if __name__ == "__main__":
    from dataset import load_dataset
    feat,target = load_dataset()
    chunk = chunking(feat,target,time_window=3)
    print(chunk[0].shape)
    print(chunk[0][-10:])
    print(chunk[1].shape)
    print(chunk[1][-10:])
