import numpy as np
import os
import re

class create_dataset():

    def __init__(self):
        self = self
        self._target = {"8":0,"14":1,"28":2}
        self.data = np.array()

    def check_path(self):
        os.chdir('/home/%s' %os.environ['USER'])
        Y = os.listdir()
        if any(n == "data" for n in Y) == False:
            print('Path to \'data\' directory not found')
            path = input('Please enter the path:')
            os.chdir(path)
            Y = os.listdir()
        else:
            os.chdir('data/')
        if any(n.startswith('SUBJ') for n in Y) == False:
            print('Directory with data not found')
        else:
            # self._DirectoriesList = [n for n if n.startswith('SUBJ')]
            self._DirectoriesList = [n.startswith('SUBJ') for n in os.listdir()]
            print(self._DirectoriesList)

    def _create_array(path):
        x = np.load_txt(z,delimiter = ',')
        re_sult = re.search('(?P<freq>\d)+Hz',z)
        f = re_sult.group('freq')
        target = self._target[f]
        t = [target for n in range(x.shape[0])]


    def create_first_array(self):

        os.chdir(self._DirectoriesList[0])

        X = self._create_array(os.listdir[0])
        self.data = X
        os.chdir('../')

    def reading(self):
        for n in self._DirectoriesList:
            os.chdir(n)
            for z in os.listdir():
                t = self._create_array(z)
                X = np.column_stack(x,t)
                self.data = np.vstack(self.data,X)
            os.chdir('../')
    def run(self):
        self.check_path()
        self.create_first_array()
        self.reading()

        np.savez('dataset.npz',self.data)


# if __name__ == "__main__":
dc = create_dataset()
dc.check_path()

#TODO Figure out how to create first array, complete creating array function
