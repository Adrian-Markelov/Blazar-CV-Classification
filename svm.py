import numpy as np 
from sklearn.svm import SVC
from extract_data import get_all_data

if __name__ == "__main__":
    clf = SVC()
    data = get_all_data()
    clf.fit(X, y)


