import numpy as np
from BakSys import load_data_path
import scipy.signal as sig


class FeatFFT(object):

    def __init__(self,threeclass=True, freq = 256,
    peaks = [[7,8,9],[13,14,15],[27,28,29]]):
        self.self = self
        self.threeclass = threeclass
        self.freq = freq
        self.peaks = peaks

    @staticmethod
    def __extractpeaks(X,peaks):
        """
        Extract peaks from data.
        """
        Y = X[:,[peaks]]
        y = np.zeros((3))
        for n in range(Y.shape[2]):
            y[n] =  np.mean(Y[:,[0],[n]])

        return np.max(y)

    def fit(self,data):

        """
        Extract frequency features from data.

        """
        #Firstly, we need to prepare from which frequencies

        X = data
        #C is number of channels
        C = X.shape[0]
        F = self.freq
        Y = np.zeros((C,F))

        #Next, we perform Fourier Transformation on data from every channel
        for n in range(C):
            Y[n] = (2*abs(np.fft.fft(sig.hamming(len(X[n]))*X[n],F))/F)

        l = self.__extractpeaks(Y,[7,8,9])    #low frequencies
        m = self.__extractpeaks(Y,[13,14,15]) #medium frequencies
        self.featFFT = np.array([l,m])
        if self.threeclass == True:
            h = self.__extractpeaks(Y,[27,28,29]) #high frequencies
            self.featFFT = np.hstack((self.featFFT,h))

    def transform(self):
        features = self.featFFT
        return features

    def fit_transform(self,data):
        self.fit(data)
        return self.featFFT

def extractmorph(data):

    """
    Extract morphological features from data.
    """

    for n in range(3):
        y = np.max(data[n])
        t = np.min(data[n])
    dif = y-t
    morphfeat = np.array((y,t,dif,s))

data = load_data_path("../subject1/sd28Hz1sec/28Hz1sec0prt2trial.csv")
FFT = FeatFFT()
# FFT.fit(data)
# features = FFT.transform()
features = FFT.fit_transform(data)
print(features)

#TODO Documentation!
