from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
import os, sys


if __name__ == "__main__":
    try : 
        # Ingestion
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        print("train, test path :", train_path, test_path)
        print("Data Ingestion Completed")
    except Exception as e : 
        logging.info('Exception occured at Data Ingestion')
        raise CustomException(e,sys)


    try : 
        # Transformation
        obj2 = DataTransformation()
        x_train, y_train, x_test, y_test = obj2.initiate_data_transformation(train_path,test_path)
        print("Data Transformation Completed")

    except Exception as e  :
        logging.info('Exception occured at Data Tranformation')
        raise CustomException(e,sys)
    

    try :
        # Training
        obj3 = ModelTrainer()
        obj3.initiate_model_training(x_train, y_train, x_test, y_test)
        print("Model Training Completed")

    except Exception as e  :
        logging.info('Exception occured at Model Training')
        raise CustomException(e,sys)