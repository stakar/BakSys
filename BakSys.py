# This file is containing a class, that perform all modules for
# Hovagim Bakardjian system that serves for feature extraction and command
# classification in SSVEP based BCI.
# Also, it has an built-in FFT features extractor

# Version 3.3.1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.stats import mode
from amuse import AMUSE
from sklearn.cross_decomposition import CCA

class BakardjianSystem(object):

    def __init__(self, path, extract=False, freq=256,
                 sep = ' ', channels = [15,23,28],threeclass = True,
                 seconds = 1):
        """
        Bakardjian System

        Bakarjian System is a class that takes EEG data and performs signal
        analysis proper to Bakardjian system.

        Parameters
        ----------

        path : str
            an input path that leads to the EEG data on which operations should be
            performed. Data should be with csv or tsv extension, with shape [n_channels,n_probes]

        sep : str
            a separator used in data that are supposed to be load

        extract : bool
            decision whether to extract comoponents or not

        freq : int
            sampling frequency of data

        sep : int
            delimiter used in data

        channels : list
            channels on which analysis is supposed to be performed

        threeclass : boolean
            decision whether perform two- or three class classification.

        seconds : int
            length of time window in seconds

        Attributes
        ----------


        References
        ----------

        Hovagim Bakardjian, Optimization of steady-state
        visual responses for robust brain-computer interfaces. 2010

        """

        self.self = self
        self.path = path
        self.extract = extract
        self.freq = freq
        self.channels = channels
        self.sep = sep
        self.threeclass = threeclass
        self.seconds = seconds

    def load_data(self):
        """
        Load data from input path and extract artifacts.

        """
        data = np.loadtxt(self.path,delimiter=self.sep)

        if self.extract == True:
            amuse = AMUSE(data,data.shape[0],1)
            data = amuse.sources

        self.data = data[self.channels,:]


    @staticmethod
    def __filtering(data,low,high,freq):

        """
        Filter the data using band-pass filter.

            Parameters
            ----------

            data : array
                Array of data, that is signal supposed to be filtered.

            low  : float
                Lower band of frequency

            high : float
                Higher band of frequency

            freq : int
                Frequency of sampling

        """
        bplowcut = low/(freq*0.5)
        bphighcut = high/(freq*0.5)
        [b,a] = sig.butter(N=3,Wn=[bplowcut,bphighcut],btype='bandpass')
        filtered = sig.filtfilt(b,a,data)

        return filtered

    def __matfilt(self,data,low,high,freq):

        """
        Filter the matrix of data using built-in band-pass filter.

            Parameters
            ----------

            data : matrix array-like, shape [n_channels,n_probes]
                Matrix of data

            low  : float
                Lower band of frequency

            high : float
                Higher band of frequency

            freq : int
                Frequency of sampling
        """
        C, P = data.shape
        result = np.zeros([C,P])

        for n in range(C):
            result[n,:] = self.__filtering(data[n,:],low,high,freq)

        return result

    def bank_of_filters(self):

        """
        Filter each channel using narrow bandpass filters.
        """

        x = self.data
        X = self.__matfilt(x,7.9,8.1,self.freq)
        Y = self.__matfilt(x,13.9,14.1,self.freq)
        self.data = np.array([X,Y])

        if self.threeclass == True:
            Z = self.__matfilt(x,27.9,28.1,self.freq)
            self.data = np.array([X,Y,Z])

    def variance_analyzer(self):

        """
        Extract energy variance of signal.
        """

        self.data = abs(self.data)

    def smoothing(self):

        """
        Smooth the data
        """

        F,C,P = self.data.shape
        X = np.zeros((F,C,P))

        for n in range(0,F):
            for i in range(C):
                x = self.data[n,[i]]
                X[n,[i]] = sig.savgol_filter(x,polyorder=2,
                                            window_length =(self.freq*self.seconds)-1,
                                            deriv=0,mode='wrap')

        self.data = X

    def integrating(self):

        """
        Integrate channels for each analyzed frequency.
        """

        data = self.data
        F,C,P = data.shape
        result = np.zeros((F,1,P))

        for n in range(F):
            for z in range(P):
                result[n,0,[z]] = np.mean(data[n,:,[z]])

        self.data = result


    def normalize(self):

        """
        Normalize the data.
        """

        F,C,P = self.data.shape
        S = np.zeros((1,C,P))

        for n in range(F):
            S += self.data[n]

        for n in range(F):
            self.data[n] = self.data[n]/S

    def run(self):
        """
        This method performs all modules from Bakardjian System.
        """
        ### 1. Firstly, load the data and extract components; those are modules 1. and 2.
        ### from original Bakardjian System
        bs.load_data()

        ### 2. Secondly, filter the data on two or three frequencies, depending on what
        ### classification type you focuse on.
        bs.bank_of_filters()

        ### 3. Extract energy band variance
        bs.variance_analyzer()

        ### 4. Smooth the data using Savitzky-GOlay 2nd order filter
        bs.smoothing()

        ### 5. Integrate channels
        bs.integrating()

        ### 6. Normalize the data
        bs.normalize()

        ### 7. Output is a data attribute.

    def extractmorph(self):

        """
        Extract morphological features from data.
        """

        bs.data.shape

        for n in range(3):
            y = np.max(bs.data[n])
            t = np.min(bs.data[n])
        dif = y-t
        self.morphfeat = np.array((y,t,dif,s))

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

    def extractFFT(self):

        """
        Extract frequency features from data.

        """
        #Firstly, we need to prepare
        freq = [[7,8,9],[13,14,15],[27,28,29]]

        X = self.data
        C = X.shape[0]   #C is number of channels
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

    def bs_classifier(self):

        """
        Built-in classifier
        """

        X = self.data.squeeze()
        C,P = X.shape
        classified = np.zeros((P))

        if self.threeclass == False:
            for n in range(P):
                dict_classes = {X[0,n]:0, X[1,n]:1}
                val_max = np.max(X[:,n])
                classified[n] = dict_classes[val_max]

        if self.threeclass == True:
            for n in range(P):
                dict_classes = {X[0,n]:0, X[1,n]:1, X[2,n]:2}
                val_max = np.max(X[:,n])
                classified[n] = dict_classes[val_max]

        print(classified)


# if __name__ is "__main__":
bs = BakardjianSystem("../subject1/sd28Hz1sec/28Hz1sec0prt2trial.csv",
                      freq = 256,channels=[15,23,28],
                      extract=True,
                      threeclass=True,
                     seconds =3)
bs.load_data()
bs.run()
print(bs.data.shape)
bs.bs_classifier()

# bs.extractFFT()
# print(bs.featFFT)

# TODO: In bank of filters, change the way of creating threeclass
# data, so it does not create new dataset, but rather just add
# Z array to dataset
# TODO: Extraction of morphological and FFT features
# TODO: Error handling
# TODO: Bakardjian classifier
