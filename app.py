import streamlit as st
from pymongo.mongo_client import MongoClient
from src import utils
import pandas as pd
from src.pipelines import prediction_pipeline
from datetime import datetime
import json, io


uri = "mongodb+srv://rajput89207:rajput89207@cluster0.q4cidjn.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

st.sidebar.title('Water Fault Detection')

if st.sidebar.button('About Creator'):
    utils.about_me()

spectra = st.file_uploader("upload file", type={"csv", "txt"})
if spectra is not None:
    spectra_df = pd.read_csv(spectra)
    obj = prediction_pipeline.PredictionPipeline()
    y_pred = obj.predict(spectra_df)
    st.write(spectra_df)
    if st.button('Predict'):
        st.write(y_pred)
        DATABASE_NAME = "WaterFault"
        time_based = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
        COLLECTION_NAME_FOR_PRED = time_based + "_pred"
        COLLECTION_NAME_FOR_TEST = time_based + "_test"
        json_record_test = list(json.loads(spectra_df.T.to_json()).values())
        json_record_pred = list(json.loads(y_pred.T.to_json()).values())
        client[DATABASE_NAME][COLLECTION_NAME_FOR_TEST].insert_many(json_record_test)
        client[DATABASE_NAME][COLLECTION_NAME_FOR_PRED].insert_many(json_record_pred)
        st.write("Predicted file is automatically uploaded in MongoDB")
        # Convert y_pred to a CSV file in bytes
        csv_data = y_pred.to_csv().encode()

        # Create a file-like object from the bytes data
        csv_file = io.BytesIO(csv_data)

        # Offer the file for download using the download button
        btn = st.download_button(
            label="Download CSV file",
            data=csv_file,
            file_name="pred.csv",
            mime="text/csv"
        )