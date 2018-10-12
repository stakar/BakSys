import numpy as np
from BakSys import BakardjianSystem as BakSys
from BakSys import load_data_path
from sklearn.cross_decomposition import CCA
from scipy.stats import mode



class BakSysCCA(object):

    def __init__(self,freq=256,channels=[15,23,28],sep=' ',extract = False,
    threeclass=True,seconds=1,target_freq = [8,14,28],k=1):

        """
        Classifier of EEG signal in BCI prepared for Bakardjian System which use
        s Canonical Correlation Analysis.
        version 3.0

        This new version takes parameters of correlation between signal and sinu
        soidal reference, as for cosinusoidal, creates tableau for learning case
        s and uses it for classyfing using KNN classifier.

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
        self.k = k

        self.bs = BakSys(extract = self.extract, freq = self.freq,
                         sep = self.sep,channels = self.channels,
                         threeclass=self.threeclass,seconds=self.seconds)

        if threeclass == True:
            self.n_corr = 3
        else:
            self.n_corr = 2

    def _reference(self,target,harmonics=2):

        """Simple reference generator, takes as an input targetted frequency,
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

        """Correlation extractor. Takes as an input signal and reference,
        then calculates canonical correlation between them. After that
        it aquires cross-correlation between cca coefficients and returns
        asolute value of it."""

        data = data.reshape(data.shape[1],1)
        reference = reference.reshape(reference.shape[0],1)
        cancor = CCA(n_components=1)
        u,v = cancor.fit_transform(data,reference)
        coef = np.corrcoef(u.T,v.T)
        return np.abs(coef[0,1])

    def _create_instance(self,x):
        n_corr = self.n_corr
        target_ref = self.target_freq
        vect = np.zeros([n_corr*2])
        X = self.bs.fit_transform(x)
        for channel in range(n_corr):
            ref = cancor._reference(target_ref[channel])
            vect[channel*2] = cancor._extract_corr(X[channel],ref[0])
            vect[channel*2+1] = cancor._extract_corr(X[channel],ref[1])
        return vect

    def fit(self,X_train,y_train):
        n_probes = X_train.shape[0]
        n_corr = self.n_corr
        target_ref = self.target_freq
        tableau = np.zeros((n_probes,n_corr*2)) #n_corr for each of sin and cos corr
        for probe in range(n_probes):
            tableau[probe] = self._create_instance(X_train[probe])
        self.tableau = tableau
        self.labels = y_train

    @staticmethod
    def _get_distance(instance1,instance2):
        return np.linalg.norm(instance1 - instance2)

    def _get_distance_tableau(self,x):
        tableau = self.tableau
        n_corr = self.n_corr
        target_ref = self.target_freq
        instance = self._create_instance(x)
        N = tableau.shape[0]
        res_tab = np.zeros(N)
        for n in range(N):
            res_tab[n] = self._get_distance(tableau[n],instance)
        return res_tab

    def predict(self,x):
        y_train = self.labels
        distance_tableau = self._get_distance_tableau(x)
        vote_table = np.zeros([self.k])
        for n in range(self.k):
            neighbour = np.min(distance_tableau)
            vote = y_train[np.where(neighbour)]
            y_train = np.delete(y_train,vote)
            distance_tableau = np.delete(distance_tableau,neighbour)
            vote_table[n] = vote
        prediction = int(mode(vote_table)[0][0])
        return prediction

    def score(self,X_test,y_test):
        predicted = np.zeros(y_test.shape)
        N_cases = X_test.shape[0]
        for case in range(N_cases):
            predicted[case] = self.predict(X_test[case])
        result = predicted[predicted == y_test].shape[0]/N_cases
        return result

if __name__ == "__main__":
    from chunking_data import load_chunked_dataset
    X,y = load_chunked_dataset(time_window = 3)

    #Comment two lines below for two-class classification
    X = X[np.where((y != 2)),:][0]
    y = y[np.where((y != 2)),:][0]

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    cancor = BakSysCCA(extract=True,seconds = 3,k=1,threeclass=False)
    cancor.fit(X_train,y_train)
    print(cancor.tableau.shape)
    print(cancor.predict(X_test[0]))
    print(cancor.score(X_test,y_test))
