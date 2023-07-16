from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sys, os
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self, x_train, y_train, x_test, y_test):
        logging.info("Model Training Started")
        try:
            models={
                "logistic":LogisticRegression(),
                "SVC":SVC(),
                "Decision Tree":DecisionTreeClassifier(),
                "adaboost":AdaBoostClassifier(),
                "Gradient":GradientBoostingClassifier(),
                "Random forest":RandomForestClassifier(),
            }
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print('Model Name : Score')
            for key, value in model_report.items():
                print(key, ':', value)
            print('\n===========================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f'Best Model Name : {best_model_name} \n accuracy Score : {best_model_score}')
            print('\n===========================================================================================\n')
            logging.info(f'Best Model Name : {best_model_name} \n accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)