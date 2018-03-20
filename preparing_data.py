
# coding: utf-8

# In[26]:


from bieg import ICAManager
import pandas as pd
import numpy as np
import os


class preping_data(object):
    
    def __init__(self,path,sec=3,hz='0',dset = np.zeros(((15,256,128))),part=1):
        self.path = path
        self.dset = dset
        self.sec = int(sec)
        self.hz = hz
        self.part = part
    def preping(self):
        file = ICAManager(input_path=self.path,method='fastica',sep=',')
        data = file.load_data()
        self.dset = file.data[256*5:256*20]
        self.dset = self.dset.reshape((((15//self.sec),128,self.sec*256)))
        return self.dset
    def writing(self):
        name_path = 'data%sHz_%s_seconds' %(self.hz,str(self.sec))
        if name_path not in os.listdir():
            os.mkdir(name_path)
        os.chdir(name_path)
        np.savetxt(fname='ssvep%sHz_sec%s_prt%s.csv' %(self.hz,str(self.sec),str(self.part)),X=self.dset[self.part])
        os.chdir('..')
    def prep_write(self):
        self.preping()
        self.writing()
        


# In[44]:


for n in range(0,15):
    predat = preping_data(path='../data/SUBJ1/SSVEP_14Hz_Trial1_SUBJ1.csv',hz='14',sec=1,part=n)
    predat.prep_write()

