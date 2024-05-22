import numpy as np
import pandas as pd
class SVM:
    def __init__(self, C=1.0, alpha=0.01):
        self.C = C              # Parámetro de regularización
        self.alpha = alpha      # Tasa de aprendizaje
        self.w = None           # Vector de pesos
        self.b = None           # Sesgo

    def fit(self, X, y, epochs=1000):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(epochs):
            # Cálculo de y_aprox y pérdida
            y_aprox = self.h(X)
            loss = self.loss(y, y_aprox)

            # Cálculo de derivadas
            dw, db = self.derivatives(X, y, y_aprox)

            # Actualización de pesos y sesgo
            self.update(dw, db)

    def h(self, X):
        return np.dot(X, self.w) + self.b

    def loss(self, y, y_aprox):
        regularizacion = 0.5 * np.linalg.norm(self.w)**2
        hinge_loss = self.C * np.sum(np.maximum(0, 1 - y * y_aprox))
        return regularizacion + hinge_loss

    def derivatives(self, X, y, y_aprox):
        n = len(y)
        dw = np.zeros_like(self.w)
        db = 0

        for i in range(n):
            if y[i] * y_aprox[i] < 1:
                dw += self.w - self.C * y[i] * X[i]
                db -= self.C * y[i]
            else:
                dw += self.w

        dw /= n
        db /= n

        return dw, db

    def update(self, dw, db):
        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.sign(self.h(X))





