# This file is containing a class, that perform all modules for
# Hovagim Bakardjian system that serves for feature extraction and command
# classification in SSVEP based BCI.
# Also, it has an built-in FFT features extractor

# Version 3.2

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
            
        twoclass : boolean
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
    def filtering(data,low,high,freq):
        
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
    
    def matfilt(self,data,low,high,freq):
        
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
            result[n,:] = self.filtering(data[n,:],low,high,freq)
            
        return result
    
    def bank_of_filters(self):
        
        x = self.data
        X = self.matfilt(x,7.9,8.1,self.freq)
        Y = self.matfilt(x,13.9,14.1,self.freq)
        self.data = np.array([X,Y])
        
        if self.threeclass == True:
            Z = self.matfilt(x,27.9,28.1,self.freq)
            self.data = np.array([X,Y,Z])
            
        
    def variance_analyzer(self):
        
        self.data = abs(self.data)
        
    def smoothing(self):
        
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
        
        data = self.data
        F,C,P = data.shape
        result = np.zeros((F,1,P))

        for n in range(F):
            for z in range(P):
                result[n,0,[z]] = np.mean(data[n,:,[z]])              
        
        self.data = result
        
    
    def normalize(self):
        
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
        
        bs.data.shape
        
        for n in range(3):
            y = np.max(bs.data[n])
            t = np.min(bs.data[n])
            dif = y-t
            res = np.array((y,t,dif,s))
            
        return res
        
    def extractFFT(self):
        
        """
        Extract frequency features from data
        """
        
        X = self.data
        if len(self.data.shape) == 2:
            C,P = X.shape
            
            X = self.data

            C,P = X.shape
            F = self.freq

            Y = np.zeros((C,F))

            Ymax = np.zeros((3))


            for n in range(C): 
                Y[n] = (2*abs(np.fft.fft(sig.hamming(len(X[n]))*X[n],
                                         self.freq))/self.freq)
                
                self.featFFT = np.array([Y[n][7:10],Y[n][13:16]])
                
                if self.threeclass == True:
                    self.featFFT = np.vstack((self.featFFT,[Y[n][27:30]]))
                    
                Y[n] = np.max(Y[n])
            return Y
        else:
            print("Don't filter the data before FFT extraction!")

if __name__ is "__main__":
    bs = BakardjianSystem("../subject1/sd14Hz3sec/14Hz3sec0prt4trial.csv",
                          freq = 256,channels=[15,23,28],
                          extract=True,
                          threeclass=True,
                         seconds =3)
    bs.load_data()
    bs.run()
    print(bs.data.shape)

# TODO: In bank of filters, change the way of creating threeclass
# data, so it does not create new dataset, but rather just add
# Z array to dataset
# TODO: Extraction of morphological and FFT features
#TODO: Error handling