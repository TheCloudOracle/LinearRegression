import numpy as np
from utils import *

class LinearRegression(object):
    """
    The goal is to find a line of best fit. How this will work, is that we will have the original function,
    in the form y=b+wx. Then, we will start off with random values for w and b, and use gradient descent
    to update the values. This will happen over n iterations. The goal is to minimize the loss function defined
    by the mean squared error (MSE). 
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weight = np.random.randn() # start with a random value for the weight
        self.bias = np.random.randn() # start with a random value for the bias


    def fit(self, X, y):
        # iterate over the epochs, and change the weight and bias accordingly
        epochs = int(input('Please enter number of epochs: '))
        print(f'[+] Training...')
        for epoch in range(epochs):
            # predicting
            y_pred = self.predict(X)
            error = self.error(y, y_pred)
            # TODO: experiment with X and X
            self.bias = self.bias - self.learning_rate * (2/len(X) * np.sum(y_pred - y))
            self.weight = self.weight - self.learning_rate * (2 / len(X) * np.mean((y_pred - y) * X))
            print(f'[?] Epoch: {epoch + 1} | Error: {error:.2f}')

    @staticmethod
    def error(true, pred):
        return 1/len(true) * (np.sum(pred - true) ** 2)
    
    def predict(self, X):
        return self.bias + self.weight*X 
    
    def get_formula(self):
        print(f'Predicted formula: {self.weight:.2f}X + {self.bias:.2f}')