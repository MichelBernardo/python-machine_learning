from numpy import * 


class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__correlation_coefficient = self.__correlation()
        self.__inclination = self.__inclination()
        self.__intercept = self.__intercept()
    
    def __correlation(self):
        covariation = cov(self.x, self.y, bias=True)[0][1]
        variance_x = var(self.x)
        variance_y = var(self.y)

        return covariation / sqrt(variance_x * variance_y)

    def __inclination(self):
        std_x = std(self.x)
        std_y = std(self.y)

        return self.__correlation_coefficient * (std_y / std_x)

    def __intercept(self):
        mean_x = mean(self.x)
        mean_y = mean(self.y)

        return mean_y - mean_x * self.__inclination

    def forecast(self, value):
        return self.__intercept + self.__inclination * value

    @property
    def correlation_coefficient(self):
        return self.__correlation_coefficient

    @property
    def inclination(self):
        return self.__inclination

    @property
    def intercept(self):
        return self.__intercept