import os
import sys
import json
import pandas as pd
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model_inference import ModelInference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_model_config, get_inference_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

inference = ModelInference('artifacts/models/churn_analysis.joblib')

def streming_inference(inference, data):
    inference.load_encoders('artifacts/encode')
    pred = inference.predict(data)
    return pred

if __name__ == '__main__':
    data = {
        "RowNumber": 1,
        "CustomerId": 15634602,
        "Firstname": "Grace",
        "Lastname": "Williams",
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
        }
    pred = streming_inference(inference, data)
    print(pred)



"""📌 Inference Types:
 Batch Inference:
   - Runs predictions on large datasets at once (e.g., daily, weekly).
   - Higher latency, not real-time.
   - Example: Predicting monthly customer churn for all users overnight.

 Streaming Inference:
   - Runs predictions continuously as new data arrives.
   - Very low latency, real-time or near real-time.
   - Example: Detecting fraudulent transactions instantly as they happen.
"""
