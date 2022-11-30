import pandas as pd
import matplotlib.pyplot as plt

# import pylab as pl
# import numpy as np
# import scipy.optimize as opt
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# %matplotlib inline


df = pd.read_csv("resources/FuelConsumption.csv")
print(df.head())  # See data sample
print(df.describe())  # Get a statistic data description (occurrences, averages and deviations)

# Sub-dataset creation
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(10))

# Show sub-dataset in a graphic
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
