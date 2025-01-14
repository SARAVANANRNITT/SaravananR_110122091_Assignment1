import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, is_training=True):

    if data is None:
        return None

    #Standardize column names for consistency.
    data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

    # Handle 'dependents'
    if 'dependents' in data.columns:
        data['dependents'] = data['dependents'].replace('3+', '3').fillna(0).astype(int)

    # Impute missing values for categorical features with the most frequent value (mode)
    categorical_cols = ['self_employed', 'gender', 'married', 'education', 'property_area']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Impute missing numerical values with the median
    numerical_cols = ['loanamount', 'loan_amount_term', 'credit_history']
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())

    # Label encode categorical features (only fit the encoder during training)
    label_encoder = LabelEncoder()
    for col in ['gender', 'married', 'education', 'self_employed', 'property_area']:
         if col in data.columns:
             if is_training:
                data[col] = label_encoder.fit_transform(data[col])
             else:
                # Handle unseen categories during prediction using try-except block
                try:
                     data[col] = label_encoder.transform(data[col])
                except ValueError:
                   data[col] = -1  # Assign a default value
    #Fill any remaining missing values after preprocessing
    data = data.fillna(0)
    return data

def train_classification_model(data):
    data = preprocess_data(data)
    if data is None:
         return None, None, None

    X = data.drop(['loan_id', 'loanamount'], axis=1) #Remove LoanID and LoanAmount
    y = (data['loanamount'] > 100).astype(int) #Simple classifier
    
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

    return best_model, None, None

def predict_loan_eligibility(model, input_df): # Changed input_data to input_df
    input_df = preprocess_data(input_df, is_training=False)
    input_df = input_df.fillna(0)

    X_columns = ['gender', 'married', 'dependents', 'education', 'self_employed',
                'applicantincome', 'coapplicantincome', 'loan_amount_term',
                'credit_history', 'property_area']

    input_df = input_df[X_columns]
    eligibility = model.predict(input_df)

    #Create output df
    output_df = pd.DataFrame({'eligibility': eligibility.tolist()})
    
    for index in input_df.index:
      if eligibility[index] == 0:
         output_df.loc[index,'max_loan_amount'] = input_df.loc[index,'applicantincome'] * 0.5
      else:
         output_df.loc[index, 'max_loan_amount'] = np.nan # Set to NaN if eligible

    return output_df

#Regression Task
def train_regression_model(data):
    data = preprocess_data(data)
    if data is None:
         return None, None, None
    #Remove rows with LoanAmount <=0 to prevent negative and zero values.
    data = data[data['loanamount']>0]

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

    return best_model, None, None

def predict_max_loan_amount(model, input_df):
    input_df = preprocess_data(input_df,is_training=False)
    input_df = input_df.fillna(0)

    X_columns = ['gender', 'married', 'dependents', 'education', 'self_employed',
                'applicantincome', 'coapplicantincome', 'loan_amount_term',
                'credit_history', 'property_area']

    input_df = input_df[X_columns]
    
    max_loan_amount = model.predict(input_df)
    
    output_df = pd.DataFrame({'max_loan_amount': max_loan_amount.tolist()})
    return output_df

def predict_min_loan_duration(model, input_df):
    input_df = preprocess_data(input_df,is_training=False)
    input_df = input_df.fillna(0)

    X_columns = ['gender', 'married', 'dependents', 'education', 'self_employed',
                'applicantincome', 'coapplicantincome', 'loan_amount_term',
                'credit_history', 'property_area']

    input_df = input_df[X_columns]
    # Assuming the model predicts Loan_Amount_Term and the target is to get minimum
    min_loan_duration = model.predict(input_df)
    
    output_df = pd.DataFrame({'min_loan_duration':min_loan_duration.tolist()})
    output_df['min_loan_duration'] = output_df['min_loan_duration'].apply(lambda x: max(x, 1) if x <= 240 else "Not Available")
    
    return output_df

if __name__ == "__main__":
    training_data_path = 'training_set.csv'
    test_data_path = 'testing_set.csv'

    training_data = load_data(training_data_path)
    testing_data = load_data(test_data_path)
    
    if training_data is not None:
      #Classification Task
        classification_model, classification_metrics, best_classification_params = train_classification_model(training_data.copy())
        print("Classification Model Training Complete")
        # print("Classification Metrics:", classification_metrics)
        # print("Best Parameters:", best_classification_params)

    if testing_data is not None and classification_model is not None:
        #Example of using prediction with input data
        sample_input_df = pd.DataFrame([testing_data.iloc[0]])
        sample_output  = predict_loan_eligibility(classification_model, sample_input_df)
        print("Example prediction: ")
        print(f"Sample Loan Eligibility: {sample_output.iloc[0,0]}")
        if not pd.isna(sample_output.iloc[0,1]):
          print(f"Sample Max Loan Amount if not eligible: {sample_output.iloc[0,1]}")
            
    if training_data is not None:
        #Regression Task
        regression_model, regression_metrics, best_regression_params = train_regression_model(training_data.copy())
        print("Regression Model Training Complete")
        # print("Regression Metrics:", regression_metrics)
        # print("Best Regression Parameters:", best_regression_params)

    if testing_data is not None and regression_model is not None:
         #Example of using regression for loan amount prediction with the input from the first record of test_data
        sample_input_df = pd.DataFrame([testing_data.iloc[0]])
        max_loan_amount_pred = predict_max_loan_amount(regression_model, sample_input_df)
        min_loan_duration_pred = predict_min_loan_duration(regression_model, sample_input_df)
        print("Regression Prediction: ")
        print(f"Predicted Maximum Loan Amount: {max_loan_amount_pred.iloc[0,0]}")
        print(f"Predicted Minimum Loan Duration: {min_loan_duration_pred.iloc[0,0]}")
