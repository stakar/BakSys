from dataset import load_dataset
from sklearn.model_selection import train_test_split
from SVM import SVM
from KNN import KNN
from BakSys import BakardjianSystem as BakSys
from chunking_data import load_chunked_datasetFFT
from CCA import BakSysCCA

# from chunking_data import load

X,y = load_chunked_datasetFFT()
y = y.ravel()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)

svm = SVM()
knn = KNN()
bs = BakSys()
cancor = BakSysCCA()

for model in [knn,svm]:
    model.fit(X_train,y_train)
    print(model.score(X_test,y_test))

for model in [bs,cancor]:
    model.fit(X_train)
    print(model.score(X_test,y_test))




# X,y = load_chunked_dataset()
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
