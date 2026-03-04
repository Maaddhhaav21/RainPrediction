import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and testing data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {

                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"]
                },

                "Random Forest": {
                    "n_estimators": [50,100,200]
                },

                "Gradient Boosting": {
                    "learning_rate":[0.01,0.05,0.1],
                    "n_estimators":[50,100,200]
                },

                "Logistic Regression": {},

                "KNN": {
                    "n_neighbors":[3,5,7,9]
                },

                "XGBoost": {
                    "learning_rate":[0.01,0.05,0.1],
                    "n_estimators":[50,100,200]
                },

                "CatBoost": {
                    "depth":[6,8,10],
                    "learning_rate":[0.01,0.05,0.1],
                    "iterations":[50,100]
                },

                "AdaBoost": {
                    "learning_rate":[0.01,0.05,0.1],
                    "n_estimators":[50,100,200]
                }

            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # best model score
            best_model_score = max(sorted(model_report.values()))

            # best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model: {best_model_name} with accuracy: {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)