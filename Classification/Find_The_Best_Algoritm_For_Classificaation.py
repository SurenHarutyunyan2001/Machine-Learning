import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    make_scorer, classification_report, confusion_matrix
)


class Find_The_Best:

    def __init__(self,x_train, x_test, y_train, y_test, *args, **kwargs) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def Find(self)-> None:
        # Basic pipeline
        pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

        # Parameters
        param_grid = [
            # SVC
            {
                'classifier': [SVC()],
                'preprocessing': [StandardScaler(), MinMaxScaler(), RobustScaler()],
                'classifier__C': [0.001, 0.1, 1, 10, 100],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10,100],
                'classifier__kernel': ['rbf'],
                'classifier__class_weight': [None, 'balanced']
             },
            
            # RandomForest
            {
                'classifier': [RandomForestClassifier()],
                'preprocessing': [None],
                'classifier__n_estimators': [10, 20, 50, 100, 150, 200, 300],
                'classifier__max_depth': [None, 5, 10, 15, 20],
                'classifier__class_weight': [None, 'balanced']
            },

            # LogisticRegression
            {
                'classifier': [LogisticRegression(max_iter=2000)],
                'preprocessing': [StandardScaler(), MinMaxScaler()],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear'],
                'classifier__class_weight': [None, 'balanced']
            },

            # KNeighbors
            {
                'classifier': [KNeighborsClassifier()],
                'preprocessing': [StandardScaler(), MinMaxScaler()],
                'classifier__n_neighbors': [2, 3, 4, 5, 7],
                'classifier__weights': ['uniform', 'distance']
            },

            # Multi-layer Perceptron classifier
            {
                'classifier': [MLPClassifier(max_iter=2000)],
                'preprocessing': [StandardScaler()],
                'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__alpha': [0.0001, 0.001, 0.01]
            },

            # GradientBoostingClassifier
            {
                'classifier': [GradientBoostingClassifier()],
                'preprocessing': [None],
                'classifier__n_estimators': [10, 20, 50, 75, 100, 150, 200],
                'classifier__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'classifier__max_depth': [3, 5, 7, 9, 12, 15]
            },

            # HistGradientBoostingClassifier
            {
                'classifier': [HistGradientBoostingClassifier()],
                'preprocessing': [None],
                'classifier__max_iter': [10, 20, 50, 75, 100, 150, 200],
                'classifier__learning_rate': [0.001, 0.01, 0.05, 0.1],
                'classifier__max_depth': [None, 5, 7, 9, 12]
            },
        ]

        # Metrics
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }

        # Save results
        results = {}

        for metric_name, scorer in scorers.items():
            print(f"\n Find the best model by '{metric_name}' ...")
            grid = GridSearchCV( pipe,
                                param_grid, 
                                cv = 5, 
                                n_jobs = -1, 
                                verbose = 1,
                                scoring = scorer, 
                                refit = True)
            
            grid.fit(self.x_train, self.y_train)
            y_pred = grid.predict(self.x_test)

            results[metric_name] = {
                'best_params': grid.best_params_,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }

            print(f"\n Best model by '{metric_name}':")
            print("Parameters:", grid.best_params_)
            print("Accuracy: {:.2f}".format(accuracy_score(self.y_test, y_pred)))
            print("Precision: {:.2f}".format(precision_score(self.y_test, y_pred)))
            print("Recall: {:.2f}".format(recall_score(self.y_test, y_pred)))
            print("F1 Score: {:.2f}".format(f1_score(self.y_test, y_pred)))
            print("\n Classification Report:\n", classification_report(self.y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

        # Comparison table
        df_results = pd.DataFrame(results).T
        print("\n Comparison table for all metrics:")
        print(df_results.drop(columns = 'best_params').round(3))

        # Model parameters for each metric
        print("\ The best parameters for each metric:")
        for metric, info in results.items():
            print(f"- {metric}: {info['best_params']}")

def main():
    # Loading data
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    obj = Find_The_Best(x_train, x_test, y_train, y_test)
    obj.Find()

    return 0

main()