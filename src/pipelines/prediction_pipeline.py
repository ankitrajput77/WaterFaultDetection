from src.logger import logging  
from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass
import pandas as pd
import sys, os

@dataclass
class PredictionPipelineConfig:
    preprocessor_path=os.path.join("artifacts", 'preprocessor.pkl')
    model_path=os.path.join("artifacts", 'model.pkl')
    features_path=os.path.join("artifacts", 'features.pkl')
    prediction_artifacts_path = os.path.join("prediction_artifacts", 'pred.csv')
    
class PredictionPipeline:
    def __init__(self):
        self.prediction_config=PredictionPipelineConfig()

    def predict(self,df_for_pred:pd.DataFrame):
        try:
            logging.info("pickle loading")
            preprocessor=load_object(file_path=self.prediction_config.preprocessor_path)
            model=load_object(file_path=self.prediction_config.model_path)
            features_used=load_object(file_path=self.prediction_config.features_path)
            logging.info("pickle loading completed")
            features_used=list(features_used)
            # col drop
            # features_used.remove('Good/Bad')
            data_point=df_for_pred[features_used]
            # preprocess
            data_point=preprocessor.transform(data_point)
            logging.info("Prediction Started")

            # data_point=data_point[:,:-1]
            pred = model.predict(data_point)
            pred = pd.DataFrame(data=pred)
            logging.info("Prediction Completed")
            pred.to_csv(self.prediction_config.prediction_artifacts_path, index=False, header=True)
            return pred
        except Exception as e:
            logging.info("error occured during prediction")
            raise CustomException(e,sys)


# test file predictions 

df=pd.read_csv(os.path.join('prediction_test_file','test.csv'))

obj = PredictionPipeline()
y_pred = obj.predict(df)

