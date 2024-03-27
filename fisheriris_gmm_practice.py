from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import random

iris = load_iris()
X = iris.data[:,[2,1]]
y = (iris.target == 0).astype(int)

some_num = 1 
while some_num <=100:
    random_number = random.randint(0,1000)
    print(f"seed: {random_number}")

    X_trian, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, random_state=random_number, stratify= y)

    n=2
    gmm=mixture.GaussianMixture(n_components=n, covariance_type='diag')
    gmm.fit(X_trian)

    prob = gmm.predict_proba(X_test)
    predict_labels =np.argmax(prob, axis=1)
    acc = np.mean(predict_labels == y_test)
    print("accuracy: ",acc)
    print(some_num,"/100")
    some_num += 1
    print(" ")