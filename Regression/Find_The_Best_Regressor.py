from sklearn.pipeline import Pipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Additional custom metrics
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def directional_accuracy(y_true, y_pred):
    return np.mean((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).astype(int))

# Scorer dictionary using make_scorer
# Note: MAPE and Directional Accuracy must be negated if used for optimization (greater_is_better=False)
scorers = {
    'r2': make_scorer(r2_score),
    'neg_mse': make_scorer(mean_squared_error, greater_is_better=False),
    'neg_mae': make_scorer(mean_absolute_error, greater_is_better=False),
    'neg_rmse': make_scorer(rmse, greater_is_better=False),
    'neg_mape': make_scorer(mape, greater_is_better=False),
    'dir_acc': make_scorer(directional_accuracy)
}

class Find_The_Best_Regressor:

    def __init__(self, x_train, x_test, y_train, y_test) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def Find(self) -> None:
        # Initial pipeline with placeholders
        pipe = Pipeline([
            ('preprocessing', StandardScaler()),
            ('feature_selection', 'passthrough'),
            ('regressor', SVR())
        ])

        # Grid search over different models and hyperparameters
        param_grid = [
            {
                'preprocessing': [StandardScaler()],
                'feature_selection': ['passthrough'],
                'regressor': [GradientBoostingRegressor()],
                'regressor__n_estimators': [10, 20, 50, 100, 150, 200, 300],
                'regressor__learning_rate': [0.001, 0.01, 0.1, 1],
                'regressor__max_depth': [3, 5, 10, 15, 20]
            },

            {
                'regressor': [SVR()],
                'preprocessing': [StandardScaler(), MinMaxScaler(), RobustScaler()],
                'regressor__C': [0.001, 0.1, 1, 10, 100],
                'regressor__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                'regressor__kernel': ['rbf'],
            },

            {
                'regressor': [RandomForestRegressor()],
                'preprocessing': [None],
                'regressor__n_estimators': [10, 20, 50, 100, 150, 200, 300],
                'regressor__max_depth': [None, 5, 10, 15, 20],
            },

            {
                'regressor': [LinearRegression()],
                'preprocessing': [StandardScaler(), MinMaxScaler(), RobustScaler()],
            },

            {
                'regressor': [KNeighborsRegressor()],
                'preprocessing': [StandardScaler(), MinMaxScaler()],
                'regressor__n_neighbors': [3, 5, 7],
                'regressor__weights': ['uniform', 'distance']
            },

            {
                'regressor': [GradientBoostingRegressor()],
                'preprocessing': [None],
                'regressor__n_estimators': [10, 20, 50, 75, 100, 150, 200],
                'regressor__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'regressor__max_depth': [3, 5, 7, 9, 12, 15]
            },

            {
                'regressor': [HistGradientBoostingRegressor()],
                'preprocessing': [None],
                'regressor__max_iter': [10, 20, 50, 75, 100, 150, 200],
                'regressor__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'regressor__max_depth': [None, 5, 7, 9, 12]
            },
        ]

        results = {}

        # Evaluate all models by different metrics
        for metric_name, scorer in scorers.items():
            print(f"\nFinding best model by '{metric_name}'...")
            grid = GridSearchCV(pipe,
                                param_grid,
                                cv = 5,
                                n_jobs = -1,
                                verbose = 0,
                                scoring = scorer,
                                refit = True)

            grid.fit(self.x_train, self.y_train)
            y_pred = grid.predict(self.x_test)

            # Store metrics for the best model found by the current scorer
            results[metric_name] = {
                'best_params': grid.best_params_,
                'r2': r2_score(self.y_test, y_pred),
                'mse': mean_squared_error(self.y_test, y_pred),
                'rmse': rmse(self.y_test, y_pred),
                'mae': mean_absolute_error(self.y_test, y_pred),
                'mape': mape(self.y_test, y_pred),
                'dir_acc': directional_accuracy(self.y_test, y_pred)
            }

            print(f"\nBest model by '{metric_name}':")
            print("Parameters:", grid.best_params_)
            print(f"R2 Score: {results[metric_name]['r2']:.3f}")
            print(f"MSE: {results[metric_name]['mse']:.3f}")
            print(f"MAE: {results[metric_name]['mae']:.3f}")
            print(f"RMSE: {results[metric_name]['rmse']:.3f}")
            print(f"MAPE: {results[metric_name]['mape']:.2f}%")
            print(f"Directional Accuracy: {results[metric_name]['dir_acc']:.2f}")

            # Plot prediction vs true values
            plt.figure(figsize = (6, 4))
            plt.plot(self.y_test.values, label = 'True')
            plt.plot(y_pred, label = 'Predicted')
            plt.title(f"Best by '{metric_name}'")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Show summary table sorted by R² score
        df_results = pd.DataFrame(results).T
        print("\n Comparison table (sorted by R²):")
        print(df_results.drop(columns = 'best_params').sort_values(by = 'r2', ascending = False).round(3))

        # Show best parameters for each scoring metric
        print("\nBest parameters for each metric:")
        for metric, info in results.items():
            print(f"-{metric}: {info['best_params']}")


# Load dataset
data = load_diabetes(as_frame = True)
x = data.data
y = np.log1p(data.target)  # log-transform the target variable

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Train models and perform grid search
model_finder = Find_The_Best_Regressor(x_train, x_test, y_train, y_test)
model_finder.Find()
