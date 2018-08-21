import numpy as np
import os
import re
from bunch import Bunch

class create_dataset():

    def __init__(self,target = {"8":0,"14":1,"28":2},freq=256):
        self = self
        self._target = target
        self.freq = freq

    def go_path(self):
        """
        Changes the directory for the one that contains data.
        """
        os.chdir('/home/%s/data/' %os.environ['USER'])
        Y = os.listdir()
        self._DirectoriesList = [n for n in Y if n.startswith('SUBJ')]

    def _create_array(self,path):
        """
        Serves for creating an array from path. In name of path is supposed to be
        an information about targeted frequency, which is extracted and added in
        form of additional column.
        """
        x = np.loadtxt(path,delimiter = ',')
        x = x[5*self.freq:20*self.freq]
        re_sult = re.search('(?P<freq>\d+)Hz',path)
        f = re_sult.group('freq')
        target = self._target[f]
        t = [target for n in range(x.shape[0])]
        return np.column_stack((x,t))


    def create_placeholder_array(self):
        """
        This method creates a placeholder in attribute data, so one can vertically
        stack the data forwards. The shape of output data is n_channels + 1
        (number of channels and 1 column for target).
        """
        os.chdir(self._DirectoriesList[0])
        X = np.loadtxt(os.listdir()[0],delimiter=',')
        _x = X.shape[1] + 1
        self.data = np.zeros(_x)
        os.chdir('../')

    def read_write(self):
        """
        This method iterates through all directories with data, reads it then
        stack every each to data attirbute.
        """
        X = self.data
        for n in self._DirectoriesList:
            os.chdir(n)
            for z in os.listdir():
                print(z)
                t = self._create_array(z)
                X = np.vstack((X,t))
            os.chdir('../')
        X = X[1:]
        self.data = X

    def run(self):
        """
        This method runs all modules of this class, then saves the data attribute as
        dataset.npz.
        """
        self.go_path()
        self.create_placeholder_array()
        self.read_write()
        np.save('dataset.npy',self.data)

def load_dataset():
    """
    This function loads created dataset.
    """
    path = '/home/%s/data/dataset.npy' %os.environ['USER']
    data = np.load(path)
    X = data[:,:128]
    y = data[:,128]
    return X,y
# if __name__ == "__main__":
#     dc = create_dataset()
#     dc.run()
#     data = load_dataset()
#     print(data.data.shape)
