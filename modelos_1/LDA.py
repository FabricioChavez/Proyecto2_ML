import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
class LDA:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vectors = None
        self.mean_total = None
        self.categories_number = len(np.unique(y))

    def Sw(self):
        SW = np.zeros((self.x.shape[1], self.x.shape[1]))
        for val in np.unique(self.y):
            c_i = self.x[self.y == val]
            u_i = np.mean(c_i, axis=0)
            SW += np.dot((c_i - u_i).T, (c_i - u_i))
        return SW

    def Sb(self):
        mean_total = np.mean(self.x, axis=0)
        self.mean_total = mean_total
        Sb = np.zeros((self.x.shape[1], self.x.shape[1]))
        for i in np.unique(self.y):
            xi = self.x[self.y == i]
            ni = xi.shape[0]
            mean_diferencia = np.mean(xi, axis=0)
            Sb += ni * np.outer(mean_diferencia - mean_total, mean_diferencia - mean_total)
        return Sb

    def solution(self):
        SB = self.Sb()
        SW = self.Sw()
        Sw_inv = np.linalg.inv(SW)
        Stotal = np.dot(Sw_inv, SB)
        values, vectors = np.linalg.eig(Stotal)
        self.vectors = vectors[:, :self.categories_number - 1]
        return values, self.vectors

    def predict(self, X_test):
        if self.vectors is None or self.mean_total is None:
            raise ValueError("The model has not been trained yet. Call 'solution' method first.")
        X_test_lda = np.dot(X_test - self.mean_total, self.vectors)
        distances = []
        for i in np.unique(self.y):
            mean_projection = np.mean(self.x[self.y == i] @ self.vectors, axis=0)
            distance = np.linalg.norm(X_test_lda - mean_projection, axis=1)
            distances.append(distance)
        distances = np.array(distances)
        predicted_labels = np.argmin(distances, axis=0)
        return predicted_labels
    






        
       
iris = datasets.load_iris()
x = iris.data
y = iris.target
class_names = iris.target_names

ld = LDA(x , y)
values , vectors = ld.solution()

print("VECTORS :",vectors)
print("VALUES :" ,values)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
results = clf.fit(x,y)

print("LDA SKLEARN TAMOS CAGDOS??",results.scalings_)

#######################################################
