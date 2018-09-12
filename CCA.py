import numpy as np
from BakSys import BakardjianSystem as BakSys
from BakSys import load_data_path
from sklearn.cross_decomposition import CCA


class BakSysCCA(object):

    def __init__(self,freq=256,channels=[15,23,28],sep=' ',extract = True,
    threeclass=True,seconds=1,target_freq = [8,14,28]):

        """
        Classifier of EEG signal in BCI prepared for Bakardjian System which use
        s Canonical Correlation Analysis.

        Attributes
        ----------

        """
        self.self = self
        self.freq = freq
        self.extract = extract
        self.channels = channels
        self.sep = sep
        self.threeclass = threeclass
        self.seconds = seconds
        self.target_freq = target_freq
        self.t = np.linspace(0,self.seconds,(self.seconds*self.freq))

    def _reference(self,target):

        t = self.t
        sin = np.sin(2*np.pi*target*t)
        cos = np.cos(2*np.pi*target*t)

        return sin,cos

    def corr_extract(self,data):

        F,P = data.shape
        tf = self.target_freq
        cancor = CCA(n_components=1)
        result = dict()

        for z,n in zip(tf,range(F)):
            tmp = data[n]
            tmp = np.array([tmp,tmp]).reshape(P,2)
            ref = np.array(self._reference(z)).reshape(P,2)
            u,v = cancor.fit_transform(tmp,ref)
            cro = np.corrcoef(u.T,v.T)[0,1]
            maxcor = np.max(np.abs(cro))
            result[maxcor] = z

        return result

    def fit(self,data):

        X = BakSys(extract = self.extract, freq = self.freq,sep = self.sep,
        channels = self.channels,threeclass=self.threeclass,
        seconds=self.seconds).fit_transform(data).squeeze()

        corr_coef = self.corr_extract(X)
        self._coef = corr_coef

    def predict(self,data):
        decision = self._coef[max(self._coef.keys())]
        return decision



if __name__ == '__main__':
    data = load_data_path('../subject1/sd14Hz1sec/14Hz1sec12prt1trial.csv')
    cca = BakSysCCA(threeclass=False,target_freq=[8,14])
    cca.fit(data)
    print(cca.predict(data))
