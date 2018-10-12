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
        version 2.0

        This new version takes parameters of correlation between signal and sinu
        soidal reference, as for cosinusoidal and classifies using given thresho
        lds, which where aquired by exploratory analysis.


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

        self.bs = BakSys(extract = self.extract, freq = self.freq,sep = self.sep,
        channels = self.channels,threeclass=self.threeclass,
        seconds=self.seconds)

    def _reference(self,target,harmonics=2):

        """ Simple reference generator, takes as an input targetted frequency,
        then generates reference signal ("perfect" one)"""

        seconds = self.seconds
        freq = self.freq
        t = np.linspace(0,seconds,(seconds*freq))
        sin = np.sin(2*np.pi*target*t)
        cos = np.cos(2*np.pi*target*t)
        ref_array = np.array([sin,cos])
        for n in range(1,harmonics):
            sin_n = np.sin(2*np.pi*target*t*n)
            cos_n = np.cos(2*np.pi*target*t*n)
            ref_array = np.vstack((ref_array,sin_n))
            ref_array = np.vstack((ref_array,cos_n))
        return ref_array

    @staticmethod
    def _extract_corr(data,reference):

        """ Correlation extractor. Takes as an input signal and reference,
        then calculates canonical correlation between them. After that
        it aquires cross-correlation between cca coefficients and returns
        asolute value of it."""

        data = data.reshape(data.shape[1],1)
        reference = reference.reshape(reference.shape[0],1)
        cancor = CCA(n_components=1)
        u,v = cancor.fit_transform(data,reference)
        coef = np.corrcoef(u.T,v.T)
        return np.abs(coef[0,1])

    @staticmethod
    def _bounds(corsin,corcos,sinb,cosb):

        """
        This method takes as an input correlation between sinusoidal and cosinusoidal
        reference and checks whether it
        """

        dec1 = (corsin > sinb[0] and corsin < sinb[1])
        dec2 = (corcos > cosb[0] and corcos < cosb[1])
        if (dec1 and dec2):
            return True
        else:
            return False

    def classify(self,data):

        if self.threeclass == False:
            alt = 0
        else:
            alt = 2

        ref8 = self._reference(8)
        ref14 = self._reference(14)

        corsin8 = self._extract_corr(data[0],ref8[0])
        corcos8 = self._extract_corr(data[0],ref8[1])

        corsin14 = self._extract_corr(data[1],ref14[0])
        corcos14 = self._extract_corr(data[1],ref14[1])

        res1 = self._bounds(corsin14,corcos14,[0.016,1],[0.025,0.05])
        res2 = self._bounds(corsin14,corcos14,[0.02,0.03],[0,0.01])

        res3 = self._bounds(corsin8,corcos8,[0,0.0055],[0.01,0.0175])
        res4 = self._bounds(corsin8,corcos8,[0.005,0.007],[0,0.005])

        if res1 or res2:
            return 1
        elif res3 or res4:
            return 0
        else:
            return alt

    def score(self,test,target):
        N = target.shape[0]
        result = np.zeros(target.shape)
        for n in range(N):
            chunk = self.bs.fit_transform(test[n])
            result[n] = self.classify(chunk)
        
        result = result[result == target].shape[0]/N
        return result

if __name__ == '__main__':
    from chunking_data import load_chunked_dataset
    X,y = load_chunked_dataset(time_window = 3)
    target = [0,1]
    tar_freq = [8,14]
    X = X[np.where((y != 2)),:][0]
    y = y[np.where((y != 2)),:][0]

    cca = BakSysCCA(extract = True,seconds = 3,threeclass=False)
    print(cca.score(X,y))

# TODO parameters changes for two class classification. So, for fuck sake,
# you need to explore further the data in order to find proper one.
