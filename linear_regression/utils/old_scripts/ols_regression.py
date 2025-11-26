import numpy as np
import matplotlib.pyplot as plt


'''
Linear and multiple regression based on the OLS (Ordinary Least Squares) method
'''

class regression_model:
    def __init__(self, X, Y, Xlabel, Ylabel, title):
        self.X = X
        self.Y = Y
        self.Xlabel = Xlabel
        self.Ylabel = Ylabel
        self.title = title

    def Plot_the_database(self):
        plt.plot(self.X, self.Y, marker='*',linestyle='None')
        plt.xlabel(self.Xlabel)
        plt.ylabel(self.Ylabel)
        plt.title(self.title)
        plt.grid(True)
        plt.show()

    def Linear_prediction(self):
        X_mean = np.mean(self.X)  # Speed vector mean
        Y_mean = np.mean(self.Y)  # Power vector mean

        Beta_1 = (np.sum(self.X*self.Y) - Y_mean*np.sum(self.X))/(np.sum(self.X**2) - X_mean*np.sum(self.X))  # Slope
        Beta_0 = Y_mean - Beta_1*X_mean  # Linear coefficient

        self.Y_simple = Beta_0 + Beta_1*self.X  # model output (prediction)

        return Beta_0, Beta_1, self.Y_simple

    def Plot_the_linear_prediction(self):
        plt.plot(self.X, self.Y_simple, marker='None',linestyle='-', color='r')
        plt.plot(self.X, self.Y, marker='*',linestyle='None')
        plt.xlabel(self.Xlabel)
        plt.ylabel(self.Ylabel)
        plt.title( self.title + '- Linear Regression')
        plt.grid(True)
        plt.show()

    def multiple_prediction(self, degrees_of_the_polynomial):
        self.X = np.array(self.X)
        self.degrees_of_the_polynomial = degrees_of_the_polynomial  # Degrees of the regression polynomial

        polynomial_matrix = np.column_stack([self.X ** i for i in range(1, self.degrees_of_the_polynomial)])
        XX = np.hstack((np.ones((len(self.X), 1)), polynomial_matrix))

        Beta = np.linalg.inv(XX.transpose()@XX)@XX.transpose()@self.Y  # Calculation of regression coefficients

        self.Y_multiple = Beta[0] + np.dot(XX[:, 1:], Beta[1:])  # model output (prediction)

        return Beta, self.Y_multiple

    def Plot_the_multiple_prediction(self):
        plt.plot(self.X, self.Y_multiple, marker='None',linestyle='-', color='r')
        plt.plot(self.X, self.Y, marker='*',linestyle='None')
        plt.xlabel(self.Xlabel)
        plt.ylabel(self.Ylabel)
        plt.title( self.title + '- Linear Regression')
        plt.grid(True)
        plt.show()