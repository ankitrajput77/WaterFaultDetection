import pickle
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score
import os, sys
import streamlit as st

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path) # extract the directory path from a file path.
        os.makedirs(dir_path, exist_ok=True) # making directory

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info('Expection occoured in Pickling of Model')
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        model_report = {}
        model_list = list(models.values())
        model_name_list = list(models.keys())
        for i in range(0, len(models)):
            model = model_list[i]
            model.fit(X_train,y_train) 
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test,y_pred)
            model_report[model_name_list[i]] = score

        return model_report
    
    except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function Utils')
        raise CustomException(e,sys)
    



def about_me():
        st.title("About the Creator")
        c1, c2 = st.columns([1,1])
        c1.markdown("""Hey! My name is **Ankit Rajput**, Mathematics and Computing student in Indian Institute Of Technology, Guwahati.
                    This app is for Water Fault Detection, I am updating this app continuously.
                    If you have any Questions or Suggestion about this app and you want to discuss it with me
                    Contact me on rajputankit72106@gmail.com""")
        c1.markdown("If you are interested :")
        c1.markdown("Github : https://github.com/ankitrajput77")
        c1.markdown("LinkedIn : https://www.linkedin.com/in/ankit-rajput892/")

        c2.image('static/Images/me.jpg')
