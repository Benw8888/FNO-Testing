# Run both spectral and FNO methods and compare times

import Burgers, Burgers_spectral

x_resolution = 2^10  # number of data points the x axis is split into
Δsamples = 2^13/resolution  # how many other samples we pick from the base 2^13 dataset
data = Burgers.get_dataloader(Δsamples)  # get data from Zongyi
_ ,test_data = data  # split data into train and test data

print(test_data)

FNO_learner = Burgers.train(input_data=data)

