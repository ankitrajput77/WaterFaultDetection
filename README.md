# Machine Learning Project

Sensor Fault Detection is the process of estimating the value or cost of a diamond based on various factors and characteristics. It involves using data analysis, statistical modeling, and machine learning techniques to predict the market price or worth of a diamond.

Diamonds are precious gemstones that are evaluated based on their unique features, known as the "Four Cs": carat, cut, color, and clarity. These factors, along with additional aspects such as depth and table, play a crucial role in determining a diamond's value.

# Requirements
- [sklearn](https://scikit-learn.org/stable/)
- [pandas](https://www.w3schools.com/python/pandas/default.asp)
- [Streamlit](https://docs.streamlit.io/)
- [seaborn](https://seaborn.pydata.org/)
- [python](https://www.python.org/)
- [mongoDB](https://www.mongodb.com/docs/)

# Installation and Usage

To install webApp, follow these steps:

Environment Setup
```
conda create -p env python==3.8
```
```
conda activate env/
```

1. Clone the repository:
```
git clone https://github.com/ankitrajput77/WaterFaultDetection.git
```

2. Navigate to the project directory:
```
cd WaterFaultDetection
```
3. Install dependencies:
```
pip install -r requirements.txt
```

# In Details
```
├──  artifacts                    - here's the model pickle and dataset files.
│    └── preprocessor.pkl  
│    └── model.pkl
│    └── features.pkl
│    └── raw.csv
│    └── train.csv
│    └── test.csv
│
│
├──  Logs  
│    └── time_format.log          - here's the specific run log files.
│ 
│
├──  notebooks  
│    └── EDA.ipynb                        - here's the EDA notebook.
│    └── test.csv                         - here's the test csv.
│    └── wafer.csv 		                    - here's the wafer csv.
│                         
│
│
├──  prediction_artifacts                  
│   └── pred.csv                          - here's the predicted values for test_data(it will generate after running prediction.py).
│
├──  prediction_test_file                  
│   └── test.csv
│
│
├── src                                   - The "src" folder, short for "source".
│   └── exception.py                      - Exception handling.
│   └── logger.py                         - log file handling.
│   └── utils.py                          - util functions.
│   └── components
│          └── data_ingestion.py          - code for data ingestion.
│          └── data_transformation.py     - code for data transformation.
│          └── model_trainer.py           - code for model training.
│   └── pipelines
│          └── prediction_pipeline.py     - code for model prediction 
│          └── training_pipeline.py       - code for training of model 
│
│
│
├── static                                - this folder contains frontend css files.
│   └── images
│   
│
│ 
├── upload_datato_db
│     └── upload.ipynb 
│ 
├── upload_data.py 
│ 
├── app.py                                - Code for webapp running.
│					
└──setup.py                               - project's metadata and configuration details
```
# Contributing
Any kind of enhancement or contribution is welcomed.

## Contact
If you have any questions, feedback, or suggestions, feel free to reach out to us at [rajputankit72106@gmail.com](mailto:rajputankit72106@gmail.com).
