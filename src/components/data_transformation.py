from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler # Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
from sklearn.pipeline import Pipeline # pipelines
from sklearn.compose import ColumnTransformer # merging
from src.exception import CustomException # exception handling
from src.logger import logging # logging function for loggers
from src.utils import save_object # util function for saving model pickle file 
from sklearn.preprocessing import FunctionTransformer # removing features
from sklearn.utils import resample # resampling from sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import sys,os
from dataclasses import dataclass #dataclasses


@dataclass
class DataTransformationConfig:
    data_transformation_config_path=os.path.join('artifacts','preprocessor.pkl')
    used_features_path=os.path.join("artifacts","features.pkl")


class DataTransformation:
    def __init__(self):
        self.data_tansformation_config=DataTransformationConfig()
    
    def __resample_data(self,df:pd.DataFrame):
        negative_class=df[df['Good/Bad']==-1]
        positive_class=df[df["Good/Bad"]==1]
        if negative_class.shape[0] > positive_class.shape[0]:
            positive_class_resample=resample(positive_class, replace=True, n_samples=len(negative_class), random_state=42)
            resampled = pd.concat([negative_class,positive_class_resample])
        else: 
            negative_class_resample=resample(negative_class, replace=True, n_samples=len(positive_class), random_state=42)
            resampled = pd.concat([negative_class_resample, positive_class])
        return resampled


    def get_data_transformation_obj(self):
        preprocessing_pipeline=Pipeline(
                steps=[
                    ("imputer",KNNImputer(n_neighbors=3)),
                    ("scaler",RobustScaler())
                ]
        )
        return preprocessing_pipeline

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info('Data Transformation initiated')
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            logging.info(f"Train DataFrame head : \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame head : \n{test_data.head().to_string()}")

            logging.info("preprocessing used")
            pre_processing_obj=self.get_data_transformation_obj()

            train_data = self.__resample_data(train_data)
            train_target=train_data["Good/Bad"]
            train_data=train_data.drop(['Good/Bad'], axis=1)
            test_target=test_data["Good/Bad"]
            test_data=test_data.drop(['Good/Bad'], axis=1)


            train_data=pre_processing_obj.fit_transform(train_data)
            test_data=pre_processing_obj.transform(test_data)
            logging.info("Preprocessing Completed")

            logging.info("Resampling Started")
            df1=pd.DataFrame(data=train_data,columns=pre_processing_obj.get_feature_names_out())
            # train_data=self.__resample_data(df1)
            # train_target=resampled_data['Good/Bad']
            # resampled_data.drop('Good/Bad', axis=1, inplace=True)
            logging.info("Resempling Done For Train")

            df2=pd.DataFrame(data=test_data, columns=pre_processing_obj.get_feature_names_out())
            # test_target=df2['Good/Bad']
            # df2.drop("Good/Bad", axis=1, inplace=True)
            logging.info("Resampling Done For Test")

            logging.info("Saving the preprocessor")
            save_object(file_path = self.data_tansformation_config.data_transformation_config_path, obj=pre_processing_obj)
            logging.info("Saving the features")
            save_object(file_path = self.data_tansformation_config.used_features_path, obj=pre_processing_obj.get_feature_names_out())
            return(
                train_data,
                train_target,
                test_data,
                test_target
            )
        
        except Exception as e:
            logging.info("Exception occured during data transformation")
            raise CustomException(e,sys)