import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import pickle

def load_model(model_path):
    # Load model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_data_test(data_test_path, time_split):
    # Load data
    data_test = pd.read_csv(data_test_path)
    data_test = data_test[data_test.time_snapshot == time_split]
  
    data_test["is_churn_one_month"] = (data_test["num_successful_orders_future_1_month"] == 0)
    print(data_test["is_churn_one_month"].value_counts(normalize=True))
    return data_test

def calculate_metrics(model, data_test):
    
    y_pred_prob = model.predict_proba(data_test)[:, 1]
    y_true = data_test["is_churn_one_month"]
    y_pred = y_pred_prob > 0.5

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return [accuracy, f1, auc, precision, recall, confusion]
def main():
    # Load model
    model_logistics = load_model("../models/churn_model_logistics.pkl")
    model_random_forest = load_model("../models/churn_model_randomforest.pkl")
    model_gradient_boosting = load_model("../models/churn_model_gradientboosting.pkl")    
    model_catboost = load_model("../models/churn_model_catboost.pkl")

    # Load data for testing
    data_test_path = "../feature-mart/customer_behavior_ecom_snapshot_FRM.csv"
    time_split = "2011-11-01"
    data_test = load_data_test(data_test_path, time_split)
    
    # Calculate metrics
    df_result = pd.DataFrame(columns=["model", "accuracy", "f1_score", "AUC", "precision", "recall", "confusion"])
 
    df_result.loc[0] = ["logistics"] + calculate_metrics(model_logistics, data_test)
    df_result.loc[1] = ["random_forest"] + calculate_metrics(model_random_forest, data_test)
    df_result.loc[2] = ["gradient_boosting"] + calculate_metrics(model_gradient_boosting, data_test)
    df_result.loc[3] = ["catboost"] + calculate_metrics(model_catboost, data_test)

    # Save result
    df_result.to_csv("../test-result/test_result_one_year.csv", index=False)

if __name__ == "__main__":
    main()