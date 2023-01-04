from config.fuel_vars import FuelVars
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# import pylab as pl
# import scipy.optimize as opt
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# %matplotlib inline

class SimpleLinearRegression():

    def __init__(self):
        self.df = pd.read_csv("resources/FuelConsumption.csv")
        self.cdf = self.df
        msk = np.random.rand(len(self.df)) < 0.8 # 80% will be used for training and the other 20% will be used for testing
        self.training_dataset = self.cdf[msk]
        self.testing_dataset = self.cdf[~msk]


    def print_data_summary(self):
        """
        Prints a data summary
        :return:
        """
        print(self.df.head())  # See data sample
        print(self.df.describe())  # Get a statistic data description (occurrences, averages and deviations)


    def show_subdataset_in_graphic(self):
        """
        Shows a sub-dataset in a graphic
        :return:
        """
        # Sub-dataset creation
        cdf = self.df[[FuelVars.ENGINE_SIZE.value, FuelVars.CYLINDERS.value, FuelVars.COMB.value, FuelVars.EMISSIONS.value]]
        print(cdf.head(10))

        viz = cdf[[FuelVars.CYLINDERS.value, FuelVars.ENGINE_SIZE.value, FuelVars.EMISSIONS.value, FuelVars.COMB.value]]
        viz.hist()
        plt.show()


    def compare_var_against_emission(self, var):
        """
        It shows a graphic comparing var against Emission
        :param var: Name of any of the defined vars in FuelVars
        :return:
        """
        try:
            fuel_var = FuelVars(var)
        except ValueError as e:
            print('Wrong variable name! \n ' + str(e))
            exit(1)

        plt.scatter(getattr(self.cdf, fuel_var.value), self.cdf.CO2EMISSIONS, color='blue') # Define X and Y axis and the graphic color
        # Defining axis labels:
        plt.xlabel(fuel_var.value)
        plt.ylabel(FuelVars.EMISSIONS.value)
        plt.show()


    def simple_linear_regression(self, var):

        try:
            fuel_var = FuelVars(var)
        except ValueError as e:
            print('Wrong variable name! \n ' + str(e))
            exit(1)

        # Training data distribution
        plt.scatter(getattr(self.training_dataset, fuel_var.value), self.training_dataset.CO2EMISSIONS, color='blue')
        plt.xlabel(fuel_var.value)
        plt.ylabel(FuelVars.EMISSIONS.value)
        plt.show()

        # Model
        regr = linear_model.LinearRegression()
        train_x = np.asanyarray(self.training_dataset[[fuel_var.value]])
        train_y = np.asanyarray(self.training_dataset[['CO2EMISSIONS']])
        regr.fit(train_x, train_y)
        print('Coefficients: ', regr.coef_)
        print('Intercept: ', regr.intercept_)

        # Show outputs
        plt.scatter(getattr(self.training_dataset, fuel_var.value), self.training_dataset.CO2EMISSIONS, color='blue')
        plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
        plt.xlabel(fuel_var.value)
        plt.ylabel(FuelVars.EMISSIONS.value)
        plt.show()

        # Evaluation. With this process we will be able to locate those areas where an improvement is needed
        test_x = np.asanyarray(self.testing_dataset[[fuel_var.value]])
        test_y = np.asanyarray(self.testing_dataset[['CO2EMISSIONS']])
        test_y_ = regr.predict(test_x)

        medium_absolute_error = np.mean(np.absolute(test_y_ - test_y))
        print(f'Error absoluto medio {medium_absolute_error}')

        mse = np.mean((test_y_ - test_y) ** 2)
        print(f'Error cuadrático medio (MSE) {mse}')

        square_r = r2_score(test_y, test_y_) # Best possible score is 1.0 and it could be negative
        print(f'R-cuadrado: {square_r}')


if __name__ == "__main__":
    simple_lineal_regression = SimpleLinearRegression()
    # simple_lineal_regression.compare_var_against_emission(FuelVars.CLASS)
    # simple_lineal_regression.compare_var_against_emission(FuelVars.COMB)
    # simple_lineal_regression.compare_var_against_emission(FuelVars.CITY)
    # simple_lineal_regression.compare_var_against_emission(FuelVars.YEAR)
    # simple_lineal_regression.compare_var_against_emission(FuelVars.FUEL_TYPE)
    # simple_lineal_regression.compare_var_against_emission(FuelVars.CYLINDERS)
    simple_lineal_regression.simple_linear_regression(FuelVars.ENGINE_SIZE)

