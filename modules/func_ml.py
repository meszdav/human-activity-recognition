import pandas as pd
import seaborn as sns
import time
import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib


# def find_na(df):
#
#     features_to_drop = []
#
#     for i in df.columns:
#
#         if df[i].isna().sum() > len(df)*0.3:
#             features_to_drop.append(i)
#
#     return features_to_drop
#
#
# def drop_features(df,features_to_drop):
#
#     df.drop(features_to_drop, axis = 1, inplace = True)
#
#     return df

def train_model(X_train, y_train, clf = "rfc"):

    if clf == "svc":
        model = SVC()

    elif clf == "rfc":
        model = RandomForestClassifier()

    model = Pipeline([("imputer", SimpleImputer(fill_value = 0)),
                     ('scaler', StandardScaler()), ("clf", model)])

    model.fit(X_train,y_train)

    return model


def save_model(name, model):
    '''Save the model to disk'''

    with open(name, 'wb') as file:
        pickle.dump(model, file)
