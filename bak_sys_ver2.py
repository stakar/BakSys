
"""
This file is containing a class, that perform all modules for 
Hovagim Bakardjian system that serves for feature extraction and command
classification in SSVEP based BCI. 
Also, it has an built-in FFT features extractor

Version 2
"""

import numpy as np
import pandas as pd
from bieg import ICAManager
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.stats import 
from sklearn.cross_decomposition import CCA


class BakardjianSystem(object):
    
    def __init__(self,path,seconds=1,components_to_exclude = [0,14,27,28,127],
                 chunk = 1,extract = False,freq = 256,classes = 'three_classes',normalize=True):
        self.self = self
        self.path = path
        self.seconds = seconds
        self.ica_file = ICAManager(input_path=self.path,method='fastica',sep=' ')
        self.components_to_exclude = components_to_exclude
        self.extract = extract
        self.freq = freq
        self.classes = classes
        self.normalize = normalize
        
    def prep_data_extract_components(self):
        
        self.ica_file.load_data()
        
        if self.extract == True:
            self.ica_file.extract_components()
            self.ica_file.exclude_ica_components(
                components_to_exclude=self.components_to_exclude)
        
        self.data = self.ica_file.data
        self.o1 = self.data[15]
        self.oz = self.data[23]
        self.o2 = self.data[28]

    @staticmethod
    def filtering(data,low,high,freq):
        
        bplowcut = low/(freq*0.5)
        bphighcut = high/(freq*0.5) 

        [b,a] = sig.butter(N=4,Wn=[bplowcut,bphighcut],btype='bandpass')

        filtered = sig.filtfilt(b,a,data)
        return filtered
    
    def go(self,low,high):
        
        """This functon replaces variance analyzer and smoother, serves as both"""
        
        electrode = np.zeros(self.o1.shape[0])
        
        for n in [self.o1,self.oz,self.o2]:
            filtered = self.filtering(n,low,high,self.freq)
            var = np.abs(filtered)
            smoothed = sig.savgol_filter(var,polyorder=2,
                                         window_length=self.seconds)
            electrode += smoothed
            
        electrode = electrode/3
        
        return electrode
            
    def normalizer(self):
        
        band8hz = [7.9,8.1]
        band14hz = [13.9,14.1]
        band28hz = [27.9,28.1]
        
        if self.classes == 'two_classes':
            
            sig_8hz = self.go(band8hz[0],band8hz[1])
            sig_14hz = self.go(band14hz[0],band14hz[1])
            
            if self.normalize == True:
                self.sig_8hz = sig_8hz/(sig_8hz+sig_14hz)
                self.sig_14hz = sig_14hz/(sig_8hz+sig_14hz)
            else:
                self.sig_8hz = sig_8hz
                self.sig_14hz = sig_14hz
            
        elif self.classes == 'three_classes':
            
            sig_8hz = self.go(band8hz[0],band8hz[1])
            sig_14hz = self.go(band14hz[0],band14hz[1])
            sig_28hz = self.go(band28hz[0],band28hz[1])
            
            if self.normalize == True:
                self.sig_8hz = sig_8hz/(sig_8hz+sig_14hz+sig_28hz)
                self.sig_14hz = sig_14hz/(sig_8hz+sig_14hz+sig_28hz)
                self.sig_28hz = sig_28hz/(sig_8hz+sig_14hz+sig_28hz)
            else:
                self.sig_8hz = sig_8hz
                self.sig_14hz = sig_14hz
                self.sig_28hz = sig_28hz
            
    def classifier(self):
        
        classified = np.zeros(self.sig_8hz.shape)
        
        if self.classes == 'two_classes':
            for n in range(self.sig_8hz.shape[0]):
                dict_classes = {self.sig_8hz[n]:0,self.sig_14hz[n]:1}
                val_max = np.max(np.array([self.sig_8hz[n],self.sig_14hz[n]]))
                classified[n] = dict_classes[val_max]
        
        elif self.classes == 'three_classes':
            for n in range(self.sig_8hz.shape[0]):
                dict_classes = {self.sig_8hz[n]:0,self.sig_14hz[n]:1,self.sig_28hz[n]:2}
                val_max = np.max(np.array([self.sig_8hz[n],self.sig_14hz[n],
                                          self.sig_28hz[n]]))
                classified[n] = dict_classes[val_max]
        
        decision = int(mode(classified)[0][0])
        
        return decision          
        
    def extract_FFT(self):
        
        if self.classes == 'two_classes':
            array_of_freq = np.zeros(((3,2,3)))
            for t,p in zip([self.o1,self.oz,self.o2],range(3)):
                result = 2*abs(np.fft.fft(sig.hamming(len(t))*t,self.freq))/self.freq

                for n,z in zip([[7,8,9],[13,14,15]],range(2)):
                    for r in range(3):
                        array_of_freq[p][z][r] = result[n[r]]
                        
            array_of_freq_max =  np.zeros((3,2))
            for z in range(3):
                for n in range(2):
                    array_of_freq_max[z][n] = np.max(array_of_freq[z][n])
            
            features = np.array([np.max(array_of_freq_max[:,[0]]),
                                 np.max(array_of_freq_max[:,[1]])])
                    
            self.features = features
        
        if self.classes == 'three_classes':
            array_of_freq = np.zeros(((3,3,3)))
            for t,p in zip([self.o1,self.oz,self.o2],range(3)):
                result = 2*abs(np.fft.fft(sig.hamming(len(t))*t,self.freq))/self.freq

                for n,z in zip([[7,8,9],[13,14,15],[27,28,29]],range(3)):
                    for r in range(3):
                        array_of_freq[p][z][r] = result[n[r]]
                        
            array_of_freq_max =  np.zeros((3,3))
            for z in range(3):
                for n in range(3):
                    array_of_freq_max[z][n] = np.max(array_of_freq[z][n])
            
            features = np.array([np.max(array_of_freq_max[:,[0]]),
                                 np.max(array_of_freq_max[:,[1]]),
                                 np.max(array_of_freq_max[:,[2]])])
                    
            self.features = features


class CCA_Classifier(object):
    
    """This class serves as classifier using Canonical Correlation Analysis
    for Bakardjian Method based system."""
    
    def __init__(self,path,seconds=3,freq = 256,classes = 'two_classes',extract = False):
        self = self
        self.seconds = seconds
        self.freq = freq
        self.classes = classes
        
        self.baksys = BakardjianSystem(path,seconds=seconds,normalize=False,classes=classes,extract=extract)
        self.baksys.prep_data_extract_components()
        self.baksys.normalizer()
        
    def reference(self,freq_ref = 8):
        
        """This function serves as creator of reference signal"""
        
        t = np.linspace(0,self.seconds,self.freq*self.seconds)
        
        reference = np.array(())
        
        sin = np.sin(2*np.pi*freq_ref*t).reshape(self.freq,self.seconds)
        cos = np.cos(2*np.pi*freq_ref*t).reshape(self.freq,self.seconds)
        
        return sin,cos
        
    def extract_correlation(self):
        
        if self.classes == 'two_classes':
            
            sig_8 = self.baksys.sig_8hz.reshape(self.freq,self.seconds)
            sig_14 = self.baksys.sig_14hz.reshape(self.freq,self.seconds)
            
            result = dict()
            
            for n in [sig_8,sig_14]:
                
                for z in [8,14]:
                    
                    sinref = self.reference(z)[0]
                    cosref = self.reference(z)[1]
                    
                    ccasin = CCA(n_components=1)
                    ccacos = CCA(n_components=1)
                    
                    ccasin.fit(n,sinref)
                    ccacos.fit(n,cosref)
                    
                    usin,vsin = ccasin.transform(n,sinref)
                    ucos,vcos = ccacos.transform(n,cosref)
                    
                    sincor = np.corrcoef(usin.T,vsin.T)[0,1]
                    coscor = np.corrcoef(ucos.T,vcos.T)[0,1]
                    
                    maxcor = abs(max([sincor,coscor]))
                    
                    result[maxcor] = z 
                
            self.corr_dict = result
        
        if self.classes == 'three_classes':
            
            sig_8 = self.baksys.sig_8hz.reshape(self.freq,self.seconds)
            sig_14 = self.baksys.sig_14hz.reshape(self.freq,self.seconds)
            sig_28 = self.baksys.sig_28hz.reshape(self.freq,self.seconds)
            
            result = dict()
            
            for n in [sig_8,sig_14,sig_28]:
                
                for z in [8,14,28]:
                    
                    sinref = self.reference(z)[0]
                    cosref = self.reference(z)[1]
                    
                    ccasin = CCA(n_components=1)
                    ccacos = CCA(n_components=1)
                    
                    ccasin.fit(n,sinref)
                    ccacos.fit(n,cosref)
                    
                    usin,vsin = ccasin.transform(n,sinref)
                    ucos,vcos = ccacos.transform(n,cosref)
                    
                    sincor = np.corrcoef(usin.T,vsin.T)[0,1]
                    coscor = np.corrcoef(ucos.T,vcos.T)[0,1]
                    
                    maxcor = abs(max([sincor,coscor]))
                    
                    result[maxcor] = z
                
            self.corr_dict = result
            
    def predict(self):
        
        corrdic = self.corr_dict
        res = corrdic[max(corrdic.keys())]
        
        return res

if __name__ == '__main__':
    bs = BakardjianSystem(path= 'data28Hz_3_seconds/ssvep28Hz_sec3_prt0.csv',seconds=3,classes='two_classes')
    bs.prep_data_extract_components()
    plt.show()
    bs.normalizer()
    print(bs.classifier())
    bs.extract_FFT()
    print(bs.features)

# if __name__ == '__main__':


#     seconds = 3


#     bs = BakardjianSystem(path='data08Hz_3_seconds/ssvep08Hz_sec3_prt0.csv',seconds=seconds,normalize=False)
#     bs.prep_data_extract_components()
#     bs.normalizer()

#     cca = CCA_Classifier(path='data14Hz_5_seconds/ssvep14Hz_sec5_prt2.csv',seconds=5,classes='two_classes',
#                         extract=True)
#     uno = cca.extract_correlation()
#     print(cca.corr_dict)
#     cca.predict()

