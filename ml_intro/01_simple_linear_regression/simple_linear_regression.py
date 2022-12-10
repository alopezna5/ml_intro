from config.fuel_vars import FuelVars
import pandas as pd
import matplotlib.pyplot as plt

# import pylab as pl
# import numpy as np
# import scipy.optimize as opt
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# %matplotlib inline

class SimpleLinearRegression():

    def __init__(self):
        self.df = pd.read_csv("resources/FuelConsumption.csv")


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

        cdf = self.df

        plt.scatter(getattr(cdf, fuel_var.value), cdf.CO2EMISSIONS, color='blue') # Define X and Y axis and the graphic color
        # Defining axis labels:
        plt.xlabel(fuel_var.value)
        plt.ylabel(FuelVars.EMISSIONS.value)
        plt.show()



if __name__ == "__main__":
    simple_lineal_regression = SimpleLinearRegression()
    simple_lineal_regression.compare_var_against_emission(FuelVars.CLASS)
    simple_lineal_regression.compare_var_against_emission(FuelVars.COMB)
    simple_lineal_regression.compare_var_against_emission(FuelVars.CITY)
    simple_lineal_regression.compare_var_against_emission(FuelVars.YEAR)
    simple_lineal_regression.compare_var_against_emission(FuelVars.FUEL_TYPE)
    simple_lineal_regression.compare_var_against_emission(FuelVars.CYLINDERS)

