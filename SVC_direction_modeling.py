from datetime import datetime
import pandas as pd 
import yfinance as yf
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt

ticker = "TSLA"
start = "2019-01-01"
end = "2025-01-01"

def download(ticker, start, end=datetime.now()):
  return yf.Ticker(ticker).history(start=start, end=end)[["Close"]]

data = download(ticker=ticker, start=start, end=end)

def generate_features(data, lags=2):
  data = data.copy()

  data["Log_Returns"] = np.log(data["Close"]/data["Close"].shift())  

  for i in range(1, lags+1):
    data[f"Return_Lag_{i}"] = data["Log_Returns"].shift(i)
    data[f"Close_ag_{i}"] = data["Close"].shift(i)

  target = np.log(data["Close"] / data["Close"].shift(-1))
  data["Target"] = np.where(target > 0, 1, -1)
  data.dropna(inplace=True)

  return data

def preprocessing(df, test_size=0.2):
  df = df.copy()
  shuffle=False
  assert shuffle == False

  x = df[df.columns.difference(["Target"])]
  assert "Target" not in x.columns

  y = df["Target"]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)

  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)

  return x_train, x_test, y_train, y_test

def run_svc_gridsearch(x_train, y_train):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10, 100, 1000],
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__gamma': ['scale', 'auto'],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(x_train, y_train)
    return grid

data = generate_features(data)
x_train, x_test, y_train, y_test = preprocessing(data)

data = generate_features(data)
x_train, x_test, y_train, y_test = preprocessing(data)

grid = run_svc_gridsearch(x_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
print("Test accuracy:", grid.score(x_test, y_test))

model = grid.best_estimator_
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

def classification_metrics(y_true, y_pred, dataset_name=""):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    print(f"{dataset_name} Accuracy: {acc:.4f}")
    print(f"{dataset_name} Confusion Matrix:\n{cm}")
    print(f"{dataset_name} Classification Report:\n{report}\n")
    return acc, cm, report

train_metrics = classification_metrics(y_train, y_train_pred, dataset_name="Train")
test_metrics = classification_metrics(y_test, y_test_pred, dataset_name="Test")


