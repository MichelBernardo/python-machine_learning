import numpy as np
import ols_regression


data = np.loadtxt('aerogerador.dat')  # loads the database

X = data[:,0]  # Speed vector
Y = data[:,1]  # Power vector
X_mean = np.mean(X)  # Speed vector mean
Y_mean = np.mean(Y)  # Power vector mean

model = ols_regression.regression_model(X, Y, 'Speed', 'Power', 'Wind turbine')
model.Plot_the_database()

Beta_0, Beta_1, Y_simple = model.Linear_prediction()
model.Plot_the_linear_prediction()

Beta, Y_multiple = model.multiple_prediction(3)
model.Plot_the_multiple_prediction()