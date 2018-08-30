import numpy as np
from features_extraction import FeatFFT


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
    X = np.zeros(4) #placeholder-array for results

    for n in classes:
        y = features[np.where(target == n)]
        n_parts = int(y.shape[0]/freq)
        for z in range(n_parts):
            part = y[(F*z):(F*(S+z))]
            tmp = FeatFFT().fit_transform(part)
            tmp = np.hstack((tmp,n))
            X = np.vstack((X,tmp))
    result = X[1:]
    return result

if __name__ == "__main__":
    from dataset import load_dataset
    feat,target = load_dataset()
    chunk = chunking(feat,target,time_window=2)
    print(chunk.shape)
    print(chunk[-10:])
