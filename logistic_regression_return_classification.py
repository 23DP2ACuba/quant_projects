import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import yfinance as yf

# -------------------- Custom Exception --------------------
class NoParametersProvidedException(Exception):
    def __init__(self):
        self.message = "No Parameters provided for grid search"
        super().__init__(self.message)

# -------------------- Data Download --------------------
def download(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)

# -------------------- Logistic Regression Wrapper --------------------
class LRClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression()

    def get_pipeline(self):
        return Pipeline([
            ("scaler", self.scaler),
            ("clf", self.classifier)
        ])

# -------------------- Training Class --------------------
class TrainClassifier(LRClassifier):
    def __init__(self, x_train, y_train, parameters=None):
        super().__init__()
        self.pipeline = self.get_pipeline()
        self.x_train, self.y_train = x_train, y_train

        if parameters is not None:
            self.param_grid = parameters
            self.block_gscv = False
        else:
            self.block_gscv = True

    def train_grid_search(self):
        try:
            if self.block_gscv:
                raise NoParametersProvidedException()

            grid = GridSearchCV(
                self.pipeline,
                self.param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )

            grid.fit(self.x_train, self.y_train)
            self.pipeline = grid.best_estimator_

            print("Best parameters:", grid.best_params_)
            print("Best CV score:", grid.best_score_)

            return self.pipeline

        except NoParametersProvidedException as e:
            print(e)
            return None

    def train(self):
        self.pipeline.fit(self.x_train, self.y_train)
        train_score = self.pipeline.score(self.x_train, self.y_train)
        print("Training score:", train_score)
        return self.pipeline

# -------------------- Feature Engineering --------------------
class CreateFeatures:
    def __init__(self, data, lags,  lookback=1, test_size=0.2, z_score_window=20, vol_window = 30):
        self.lags = lags
        self.data = data.copy()
        self.lookback = lookback
        self.ts = test_size
        self.window = z_score_window
        self.vol_window = vol_window

    def _create_features(self):
        self.data["LogReturn"] = np.log(self.data["Close"] / self.data["Close"].shift())

        for i in range(1, self.lags + 1):
            self.data[f"Lag_{i}"] = self.data["LogReturn"].shift(i)

        ma = self.data["Close"].rolling(window=self.window).mean()
        std = data["Close"].rolling(window=self.window).std()
        self.data["Z-Score"] = (self.data["Close"] - ma) / std

        returns = (self.data.Close - self.data.Open) / self.data.Open
        self.data["Volatility"] = returns.rolling(window=self.vol_window).std()
        self.data["HL_pct"] = ((self.data.High - self.data.Low) / self.data.High)

        tmp_data = self.data["Close"].pct_change().shift(-self.lookback)
        self.data["Target"] = np.where(tmp_data > 0, 1, -1)
        
        self.data.dropna(inplace=True)

    def split(self):
        x = self.data[self.data.columns.difference(["Target"])]
        y = self.data["Target"]
        return train_test_split(x, y, test_size=self.ts, shuffle=False)

    def __call__(self):
        self._create_features()
        return self.split()

# -------------------- Main --------------------
if __name__ == "__main__":
    lags = 3
    lookback = 1
    ticker = "TSLA"
    start = "2020-01-01"
    end = "2025-01-01"
    test_size = 0.2

    param_grid = {
        "clf__penalty": ["l1", "l2", "elasticnet", None],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__solver": ["lbfgs"],
        "clf__max_iter": [200],
        "clf__class_weight": [None, "balanced"],
    }

    data = download(ticker=ticker, start=start, end=end)
    x_train, x_test, y_train, y_test = CreateFeatures(data, lags, lookback, test_size)()

    trainer = TrainClassifier(x_train=x_train, y_train=y_train, parameters=param_grid)
    model = trainer.train_grid_search() or trainer.train()

    # --- Evaluate on test set ---
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test Results ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
