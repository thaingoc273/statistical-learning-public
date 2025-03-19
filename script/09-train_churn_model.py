from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
import pickle
from sklearn.model_selection import GridSearchCV

def setup_pandas_options():
    """Configure pandas options and warnings"""
    pd.set_option('mode.chained_assignment', None)
    pd.set_option('future.no_silent_downcasting', True)
    warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(data_path):
    # Load data
    data = pd.read_csv(data_path)

    max_date = data['time_snapshot'].max()
    
    # Get data for training and testing
    data = data[data.time_snapshot < max_date]
    return data

def data_split(data, time_split):
    # Split data
    data_train = data[data['time_snapshot'] < time_split]
    data_test = data[data['time_snapshot'] >= time_split]
    return data_train, data_test

def preprocess_data(data):
    # Create target variable
    data["is_churn_one_month"] = data.num_successful_orders_future_1_month == 0

    return data

def training_pipeline(data_train, selected_features, target):

    # Define a ColumnTransformer to select specific features
    feature_selector = ColumnTransformer(transformers =[('select_features', 'passthrough', selected_features)])

    # Define a pipeline to preprocess the data and train the model
    pipeline = Pipeline([
    ('feature_selection', feature_selector),  # Select only the chosen features
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Impute missing values with 0
    ('normalizer', StandardScaler()),  # Normalize the data
    ('classifier', RandomForestClassifier())  # Train model
    ])

    #Grid search for hyperparameters
    param_grid = {'classifier__n_estimators': [ 300, 500],   # Number of trees
                    'classifier__max_depth': [4, 6, 8]}  # Depth of the trees

    # Grid search for hyperparameters
    # param_grid = {'classifier__n_estimators': [100, 300, 500],   # Number of trees
    #                 'classifier__learning_rate': [0.01, 0.1, 1]}#,  # Learning rate
    #                 #'classifier__depth': [4,  8, 10]}  # Depth of the trees

    # Create a GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(data_train, data_train[target])

    model = grid_search.best_estimator_
    

    #score = pipeline.score(data_train[selected_features], data_train[target])

    return model

def main():

    # Setup pandas options
    setup_pandas_options()

    # Load data
    data_path = 'd:/code/data/customer_behavior_ecom_snapshot_FRM_two_years.csv'
    data = load_data(data_path)

    # Split data
    time_split = '2011-11-01'
    data_train, data_test = data_split(data, time_split)

    # Preprocess data
    data_train = preprocess_data(data_train)
    data_test = preprocess_data(data_test)


    # Selected features
    selected_features = ['customer_id', 
            'total_successful_amount_past_5_month', 'num_successful_orders_past_5_month',
            'total_successful_amount_past_4_month', 'num_successful_orders_past_4_month',
            'total_successful_amount_past_3_month', 'num_successful_orders_past_3_month', 
            'total_successful_amount_past_2_month', 'num_successful_orders_past_2_month',
            'total_successful_amount_past_1_month', 'num_successful_orders_past_1_month',
            'recency', 'frequency', 'monetary', 'FrequencyScore', 'MonetaryScore', 'RecencyScore']
    target = 'is_churn_one_month'

    # Train model

    pipeline = training_pipeline(data_train, selected_features, target)
    print('Model trained accuracy: ', pipeline.score(data_train, data_train[target]))
    print('Model test accuracy: ', pipeline.score(data_test, data_test[target]))
    # print(pipeline.predict_proba(data_test))
    # Save the pipeline model
    model_path = '../models/churn_model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    filename = '../models/churn_model.pkl'
    with open(filename, 'rb') as f:
        churn_model = pickle.load(f)
    print(churn_model)
    print('Model loaded accuracy: ', churn_model.score(data_train, data_train[target]))
if __name__ == '__main__':
    main()