import logging
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pickle
import os
import sys

# Adding path for common functions
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)
from src.common import *

logger = logging.getLogger()
# set logging level as INFO 
logger.setLevel(logging.INFO)

# Merge Dehli and New Delhi
def newd(x):
    if x == 'New Delhi':
        return 'Delhi'
    else:
        return x


def get_X_y_data():
    train_data_file = os.path.join(DATA_DIR, "Data_Train.xlsx")
    train_data = pd.read_excel(train_data_file)

    train_data['Destination'] = train_data['Destination'].apply(newd)

    # Make day and month columns as Datetime columns.
    if "Date_of_Journey" in train_data.columns:
        train_data['Journey_day'] = pd.to_datetime(
            train_data['Date_of_Journey'], format='%d/%m/%Y').dt.day
        train_data['Journey_month'] = pd.to_datetime(
            train_data['Date_of_Journey'], format='%d/%m/%Y').dt.month
        train_data["Day_of_Week"] = pd.to_datetime(
            train_data['Date_of_Journey'], format='%d/%m/%Y').dt.day_of_week
        train_data["Day_of_Year"] = pd.to_datetime(
            train_data['Date_of_Journey'], format='%d/%m/%Y').dt.day_of_year

        train_data.drop('Date_of_Journey', inplace=True, axis=1)

    if "Dep_Time" in train_data.columns:
        train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
        train_data['Dep_min'] = pd.to_datetime(
            train_data['Dep_Time']).dt.minute
        train_data.drop('Dep_Time', axis=1, inplace=True)

    if "Arrival_Time" in train_data.columns:
        train_data['Arrival_hour'] = pd.to_datetime(
            train_data['Arrival_Time']).dt.hour
        train_data['Arrival_min'] = pd.to_datetime(
            train_data['Arrival_Time']).dt.minute
        train_data.drop('Arrival_Time', axis=1, inplace=True)

    # Get information on duration
    if "Duration" in train_data.columns:
        duration = list(train_data['Duration'])
        for i in range(len(duration)):
            if len(duration[i].split()) != 2:
                if 'h' in duration[i]:
                    duration[i] = duration[i] + ' 0m'
                else:
                    duration[i] = '0h ' + duration[i]

        duration_hour = []
        duration_min = []
        for i in duration:
            h, m = i.split()
            duration_hour.append(int(h[:-1]))
            duration_min.append(int(m[:-1]))

        train_data['Duration_hours'] = duration_hour
        train_data['Duration_mins'] = duration_min

        train_data.drop('Duration', axis=1, inplace=True)

    # Create dummy columns out of the Airline column.
    airline = train_data[['Airline']]
    airline = pd.get_dummies(airline, drop_first=True)

    source = train_data[['Source']]
    source = pd.get_dummies(source, drop_first=True)

    destination = train_data[['Destination']]
    destination = pd.get_dummies(destination, drop_first=True)

    if "Route" in train_data.columns and "Additional_Info" in train_data.columns:
        train_data.drop(['Route', 'Additional_Info'], inplace=True, axis=1)

    # acc to the data, price is directly prop to the no. of stops
    train_data['Total_Stops'].replace(
        {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}, inplace=True)

    # Get all the data for trianing
    data_train = pd.concat([train_data, airline, source, destination], axis=1)
    data_train.dropna(inplace=True)
    try:
        data_train.drop(['Airline', 'Source', 'Destination'],
                        axis=1, inplace=True)
    except:
        pass

    # Taking out train data.
    X = data_train.drop('Price', axis=1)
    y = data_train['Price']

    try:
        train_data.drop(['Airline', 'Source', 'Destination'],
                        axis=1, inplace=True)
    except:
        pass

    return X, y


def train_model(X, y):
    logging.info("Training the model")
    reg = ExtraTreesRegressor()
    reg.fit(X, y)
    list_feat_import = [x for x in reg.feature_importances_]
    list_feat_import = [round(x/min(list_feat_import))
                        for x in list_feat_import]

    feat_importances = pd.Series(reg.feature_importances_, index=X.columns)
    feat_importances.nlargest(20).plot(kind='barh')

    # Here we are using RandomizedSearchCV
    # Randomly tries out combinations and sees which one is the best out of them.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 15, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5, 10]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    # Random search of parameters, using 5 fold cross validation, search across 100 different combinations
    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
                                   scoring='neg_mean_squared_error', n_iter=10, cv=5,
                                   verbose=0, random_state=42, n_jobs=1)
    rf_random.fit(X_train, y_train)

    model_file_name = os.path.join(MODEL_RESSULTS_DIR, 'flight_rf.pkl')
    file = open(model_file_name, 'wb')
    pickle.dump(rf_random, file)
    logging.info(f"Model saved to {file}")
    return rf_random, [X_train, X_test, y_train, y_test]


def main():
    X, y = get_X_y_data()
    model, data = train_model(X, y)


if __name__ == "__main__":
    main()
