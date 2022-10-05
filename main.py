# Import modules
import pandas as pd
import numpy as np
from sklearn import linear_model
from regression import Regressors

# Create a dataset from the CSV file
data = pd.read_csv("data.csv", sep=";")
data = data.drop(columns="Date")

# We are going to train model to predict meteorology parameters according to data from N previous days

# As sklearn doesn't allow passing 3D arrays into the fit() method, we'll use each parameter separately
temperature = np.array(data["Temperature"])
humidity = np.array(data["Humidity"])
wind_speed = np.array(data["Wind Speed"])
pressure = np.array(data["Pressure"])

# The last index of dataframe, used to train the model
train_end = int(len(data) * 0.8)  # normally, 80% of dataset is used to train and 20% to test


# Generates a list of all possible arrays of n consequent parameters
def group_arrays(array, n):
    groups = []

    for i in range(len(array) - n):
        groups.append(array[i:i + n])

    return groups


# trains model for given dataset grouped by n days
def train_model(dataset, n):
    dataset = to_standard_format(dataset)  # translate into standard format (from 0 to 1 range)
    grouped = group_arrays(dataset, n)  # group dataset by n days

    # prepare datasets for training the model
    x = grouped[:len(dataset) - n]
    target = dataset[n:]

    # build and train the model
    regression_model = linear_model.LinearRegression()
    regression_model.fit(x, target)

    return regression_model


# tests the model and returns its MAE
def test_model(dataset, n, split, value_type="R"):
    standard_dataset = to_standard_format(dataset, value_type)
    grouped = group_arrays(standard_dataset, n)
    regression_model = train_model(standard_dataset, n)
    prediction = to_source_format(regression_model.predict(grouped[split - n:]), value_type, dataset)

    return mean_abs_error(dataset[split:], prediction)


def mean_abs_error(real, predicted):
    difference = abs(real - predicted)
    return sum(difference) / len(difference)


def zero_theory_mae(dataset, split):
    real = dataset[split:len(dataset)]
    predicted = dataset[split - 1:len(dataset) - 1]

    return mean_abs_error(real, predicted)


# finds optimal value for n to pass it to test_model() function
def optimal_n(array, split, value_type="R", limit=120):
    opt_n = 1
    mae = zero_theory_mae(array, split)

    for n in range(1, limit):
        mae_n = test_model(array, n, split, value_type)
        if mae_n < mae:
            opt_n = n
            mae = mae_n

    return opt_n


def to_standard_format(dataset, value_type="R"):
    if value_type == "percent":
        return dataset / 100
    elif value_type == "positive":
        return dataset / max(dataset)
    else:
        data_range = max(dataset) - min(dataset)
        return (dataset - min(dataset)) / data_range


def to_source_format(standardised, value_type="percent", source=None):
    if value_type == "percent":
        return standardised * 100
    elif value_type == "positive":
        return standardised * max(source)
    else:
        data_range = max(source) - min(source)
        return min(source) + standardised * data_range


def predict_tomorrow(model, dataset, value_type="R"):
    standard_dataset = to_standard_format(dataset, value_type)
    prediction = to_source_format(model.predict(standard_dataset), value_type, dataset)
    return prediction


MAE_temp = test_model(temperature, optimal_n(temperature, train_end), train_end, "R")
MAE_humidity = test_model(humidity, optimal_n(humidity, train_end), train_end, "percent")
MAE_wind = test_model(wind_speed, optimal_n(wind_speed, train_end), train_end, "positive")
MAE_pressure = test_model(pressure, optimal_n(pressure, train_end), train_end, "R")

print(MAE_temp, MAE_humidity, MAE_wind, MAE_pressure)

zero_t = zero_theory_mae(temperature, train_end)
zero_h = zero_theory_mae(humidity, train_end)
zero_w = zero_theory_mae(wind_speed, train_end)
zero_p = zero_theory_mae(pressure, train_end)

print(zero_t, zero_h, zero_w, zero_p)
