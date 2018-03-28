
"""
This file is containing a class, that perform all modules for 
Hovagim Bakardjian system that serves for feature extraction and command
classification in SSVEP based BCI. 

"""

import aseegg as ag
import numpy as np
import pandas as pd
from bieg import ICAManager
import matplotlib.pyplot as plt
import math
import scipy.signal as sig
from scipy.stats import mode

#good set of components to exclude: 0,14,27,28,127

class BakardjianSystem(object):
    
    def __init__(self,path,seconds,components_to_exclude = [0,14,27,28,127],s_08Hz=np.zeros(1),
                s_14Hz = np.zeros(1),s_28Hz = np.zeros(1),es_08Hz = np.zeros(1),
                es_14Hz = np.zeros(1),es_28Hz = np.zeros(1),dset = np.zeros(((15,256,128))),chunk = 1):
        self.path = path
        self.seconds = seconds
        self.ica_file = ICAManager(input_path=self.path,method='fastica',sep=' ')
        self.components_to_exclude = components_to_exclude
        self.s_08Hz = s_08Hz
        self.s_14Hz = s_14Hz
        self.s_28Hz = s_28Hz
        self.es_08Hz = es_08Hz
        self.es_14Hz = es_14Hz
        self.es_28Hz = es_28Hz
        
    #Module 1: Blind Source Separation/ICA 
    def bss_ica(self):
        self.ica_file.load_data()
        #self.ica_file.extract_components()
        #self.ica_file.exclude_ica_components(components_to_exclude=self.components_to_exclude)
        return self.ica_file
    
    #Module 2: Narrow-band filters
        
    def bank_of_filters(self):
    
        ica_file = self.ica_file
        
        electrode_O1 = ica_file.data[15]
        electrode_0z = ica_file.data[23]
        electrode_O2 = ica_file.data[28]

        prefiltered = [electrode_0z,electrode_O1,electrode_O2]

        #Signals filtered:

        self.s_08Hz = np.zeros((3,len(electrode_O1)))
        self.s_14Hz = np.zeros((3,len(electrode_O1)))
        self.s_28Hz = np.zeros((3,len(electrode_O1)))

        for n in range(0,3):
            signal_filtered_08Hz = ag.pasmowoprzepustowy(prefiltered[n],czestOdciecia1=7.9,czestOdciecia2=8.1,czestProbkowania=256)
            signal_filtered_14Hz = ag.pasmowoprzepustowy(prefiltered[n],czestOdciecia1=13.9,czestOdciecia2=14.1,czestProbkowania=256)
            signal_filtered_28Hz = ag.pasmowoprzepustowy(prefiltered[n],czestOdciecia1=27.9,czestOdciecia2=28.1,czestProbkowania=256)
            self.s_08Hz[n] = signal_filtered_08Hz
            self.s_14Hz[n] = signal_filtered_14Hz
            self.s_28Hz[n] = signal_filtered_28Hz
            
        return self.s_08Hz, self.s_14Hz, self.s_28Hz
        
    #Module3: Variance analyzer
        
    def variance_analyzer(self):
        for z in [self.s_08Hz,self.s_14Hz,self.s_28Hz]:
            for n in range(0,z.shape[0]):
                z[n] = np.square(z[n])
                z[n] = np.sqrt(z[n])
        return self.s_08Hz, self.s_14Hz, self.s_28Hz
    
    #Module 4: Smoothing procedure

    def smoother(self):        
        for z in [self.s_08Hz,self.s_14Hz,self.s_28Hz]:
            for n in range(0,z.shape[0]):
                z[n] = sig.savgol_filter(z[n],polyorder=2,window_length=self.seconds)
        return self.s_08Hz,self.s_14Hz,self.s_28Hz

    #Module5: Integrator
    
    def integrator(self):
        self.s_08Hz = np.divide((self.s_08Hz[0] + self.s_08Hz[1] + self.s_08Hz[2]),self.s_08Hz.shape[0])
        self.s_14Hz = np.divide((self.s_14Hz[0] + self.s_14Hz[1] + self.s_14Hz[2]),self.s_14Hz.shape[0])
        self.s_28Hz = np.divide((self.s_28Hz[0] + self.s_28Hz[1] + self.s_28Hz[2]),self.s_28Hz.shape[0]) 
        return self.s_08Hz,self.s_14Hz,self.s_28Hz
    
    #Module6: Normalization
    
    def normalizer(self):
        
        sum_of_signals = self.s_08Hz+self.s_14Hz+self.s_28Hz
        self.s_08Hz = self.s_08Hz/sum_of_signals
        self.s_14Hz = self.s_14Hz/sum_of_signals
        self.s_28Hz = self.s_28Hz/sum_of_signals
        return self.s_08Hz,self.s_14Hz,self.s_28Hz

    #Bakardjian system, all functions at once
    
    def bak_class(self):
        self.bak_system()
        c_ = np.zeros(self.s_08Hz.shape[0])
        for n in range(len(c_)):
            dict_classes = {self.s_08Hz[n]:0,self.s_14Hz[n]:1,self.s_28Hz[n]:2}
            c_[n] = max(np.array([self.s_08Hz[n],self.s_14Hz[n],self.s_28Hz[n]]))
            c_[n] = dict_classes[c_[n]]
        return c_
    
    def bak_system(self):
        self.bss_ica()
        self.bank_of_filters()
        self.variance_analyzer()
        self.smoother()
        self.integrator()
        self.normalizer()
        return self.s_08Hz,self.s_14Hz,self.s_28Hz

    