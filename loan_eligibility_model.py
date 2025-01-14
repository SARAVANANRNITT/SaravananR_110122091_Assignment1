import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully:", filepath)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, is_training=True):
    if data is None:
        print("Error: No data to preprocess")
        return None

    try:
        # Standardize column names for consistency
        data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
        print("Column names standardized")

        # Handle 'dependents'
        if 'dependents' in data.columns:
            data['dependents'] = data['dependents'].replace('3+', '3').fillna(0).astype(int)
            print("Dependents column processed")

        # Impute missing values for categorical features with the most frequent value (mode)
        categorical_cols = ['self_employed', 'gender', 'married', 'education', 'property_area']
        for col in categorical_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
        print("Missing categorical values imputed")

        # Impute missing numerical values with the median
        numerical_cols = ['loanamount', 'loan_amount_term', 'credit_history']
        for col in numerical_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        print("Missing numerical values imputed")

        # Label encode categorical features (only fit the encoder during training)
        label_encoder = LabelEncoder()
        for col in ['gender', 'married', 'education', 'self_employed', 'property_area']:
            if col in data.columns:
                if is_training:
                    data[col] = label_encoder.fit_transform(data[col])
                    print(f"Label encoded (training) '{col}'")
                else:
                    # Handle unseen categories during prediction using try-except block
                    try:
                        data[col] = label_encoder.transform(data[col])
                        print(f"Label encoded (prediction) '{col}'")
                    except ValueError:
                        data[col] = -1
                        print(f"Unknown value in '{col}' set to -1")

        #Fill any remaining missing values after preprocessing
        data = data.fillna(0)


        # Convert all columns to numeric (even if already numbers) and handle NaNs/inf
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        print("All Columns Converted to numeric")

        data = data.replace([np.inf, -np.inf], 0).fillna(0) # Handle infinite or nan values.
        print("NaN and infinite values handled") #Log
        
        #Check for any NaNs
        if data.isnull().values.any():
            print("Error: NaN values found after preprocessing")
            return None

        print("Data preprocessing complete")
        return data
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def train_classification_model(data):
    data = preprocess_data(data)
    if data is None:
        return None, None, None
    try:
        X = data.drop(['loan_id'], axis=1)
        y = (data['loanamount'] > 100).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 10],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split':[2, 4, 6]
        }
        model = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        print("Classification Model trained")
        return best_model, metrics, grid_search.best_params_
    except Exception as e:
        print(f"Error training classification model: {e}")
        return None, None, None

def predict_loan_eligibility(model, input_data):
    try:
      if not isinstance(input_data, dict):
          print(f"Error: input data is not in dictionary format but is of type: {type(input_data)}")
          return None, None

      input_df = pd.DataFrame([input_data])
      input_df = preprocess_data(input_df, is_training=False)

      if input_df is None:
          print(f"Error: Dataframe is null after preprocessing")
          return None, None

      X_columns = ['gender', 'married', 'dependents', 'education', 'self_employed',
                'applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term',
                'credit_history', 'property_area']
      input_df = input_df[X_columns]
      eligibility = model.predict(input_df)[0]

      if eligibility == 0:
            max_loan_amount = input_data['applicantincome'] * 0.5
            print("Loan not Eligible")
            return "Not Eligible", max_loan_amount
      else:
           print("Loan Eligible")
           return "Eligible", None
    except Exception as e:
        print(f"Error in eligibility prediction: {e}")
        return None, None

#Regression Task
def train_regression_model(data):
    data = preprocess_data(data)
    if data is None:
        return None, None, None
    try:
        data = data[data['loanamount'] > 0]

        X = data.drop(['loan_id','loanamount'], axis=1)
        y = data['loanamount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 10],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split':[2, 4, 6]
        }
        model = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
    
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        print("Regression Model trained")
        return best_model, metrics, grid_search.best_params_
    except Exception as e:
        print(f"Error during regression model training: {e}")
        return None, None, None

def predict_max_loan_amount(model, input_data):
    try:
        if not isinstance(input_data, dict):
          print(f"Error: input data is not in dictionary format but is of type: {type(input_data)}")
          return None
        input_df = pd.DataFrame([input_data])
        input_df = preprocess_data(input_df,is_training=False)
        if input_df is None:
            print(f"Error: Input data is not able to preprocessed")
            return None
        X_columns = ['gender', 'married', 'dependents', 'education', 'self_employed',
                    'applicantincome', 'coapplicantincome','loanamount','loan_amount_term',
                    'credit_history', 'property_area']

        input_df = input_df[X_columns]
        max_loan_amount = model.predict(input_df)[0]
        print("Predicted max loan amount",max_loan_amount)
        return max_loan_amount
    except Exception as e:
         print(f"Error predicting max loan amount: {e}")
         return None

def predict_min_loan_duration(model, input_data):
    try:
        if not isinstance(input_data, dict):
          print(f"Error: input data is not in dictionary format but is of type: {type(input_data)}")
          return None

        input_df = pd.DataFrame([input_data])
        input_df = preprocess_data(input_df,is_training=False)
        if input_df is None:
             print(f"Error: Input data is not able to preprocessed")
             return None

        X_columns = ['gender', 'married', 'dependents', 'education', 'self_employed',
                'applicantincome', 'coapplicantincome','loanamount', 'loan_amount_term',
                'credit_history', 'property_area']
        input_df = input_df[X_columns]
        min_loan_duration = model.predict(input_df)[0]

        print(f"Predicted Min Loan Duration: {min_loan_duration}")
        return max(min_loan_duration, 1) if min_loan_duration <= 240 else "Not Available"
    except Exception as e:
        print(f"Error in predicting min loan duration: {e}")
        return None

if __name__ == "__main__":
    training_data_path = 'training_set.csv'
    test_data_path = 'testing_set.csv'

    training_data = load_data(training_data_path)
    testing_data = load_data(test_data_path)
    
    if training_data is not None:
      #Classification Task
        classification_model, classification_metrics, best_classification_params = train_classification_model(training_data.copy())
        print("Classification Model Training Complete")
        print("Classification Metrics:", classification_metrics)
        print("Best Parameters:", best_classification_params)

    if testing_data is not None and classification_model is not None:
        #Example of using prediction with input data
        sample_input_data = testing_data.iloc[0].to_dict()
        sample_eligibility, sample_max_amount  = predict_loan_eligibility(classification_model, sample_input_data)
        print("Example prediction: ")
        print(f"Sample Loan Eligibility: {sample_eligibility}")
        if sample_max_amount:
            print(f"Sample Max Loan Amount if not eligible: {sample_max_amount}")
            
    if training_data is not None:
        #Regression Task
        regression_model, regression_metrics, best_regression_params = train_regression_model(training_data.copy())
        print("Regression Model Training Complete")
        print("Regression Metrics:", regression_metrics)
        print("Best Regression Parameters:", best_regression_params)

    if testing_data is not None and regression_model is not None:
         #Example of using regression for loan amount prediction with the input from the first record of test_data
        sample_input_data = testing_data.iloc[0].to_dict()
        max_loan_amount_pred = predict_max_loan_amount(regression_model, sample_input_data)
        min_loan_duration_pred = predict_min_loan_duration(regression_model, sample_input_data)
        print("Regression Prediction: ")
        print(f"Predicted Maximum Loan Amount: {max_loan_amount_pred}")
        print(f"Predicted Minimum Loan Duration: {min_loan_duration_pred}")