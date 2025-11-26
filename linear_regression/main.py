from numpy import *
from src.LinearRegression import LinearRegression


def main():
    x = array([1,2,3,4,5])
    y = array([2,4,6,8,10])

    lr = LinearRegression(x,y)
    forecast = lr.forecast(6)
    print(f'The value predicted by the regression is {forecast}.')

if __name__ == '__main__':
    main()