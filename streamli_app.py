import streamlit as st
import pandas as pd
import loan_eligibility_model as lem  # Import your model code


st.title("Loan Eligibility and Recommendation App")

# Input fields (same as your Tkinter app)
fields = ['Gender', 'Married', 'Dependents', 'Education',
          'Self_Employed', 'ApplicantIncome',
          'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term',
          'Credit_History', 'property_Area']
entries = {}


with st.form("loan_form"):

  for field in fields:
      entries[field] = st.text_input(field)

  submitted = st.form_submit_button("Check Eligibility and Get Recommendation")

if submitted:
    try:
            # Load data and train the model
        training_data_path = 'training_set.csv'
        training_data = lem.load_data(training_data_path)
        if training_data is None:
                 st.error("Unable to load training data")
                 st.stop()

        classification_model,_, _ = lem.train_classification_model(training_data.copy())
        if classification_model is None:
                 st.error("Model training failed")
                 st.stop()
        model,_,_ = lem.train_regression_model(training_data.copy())
        if model is None:
            raise ValueError("Model training failed")


            # Validate and transform user inputs
        input_data = {
                    'Gender': 1 if entries['Gender'].strip().lower() in ['male', '1'] else 0,
                    'Married': 1 if entries['Married'].get().strip().lower() in ['yes', '1'] else 0,
                    'Dependents': int(entries['Dependents'].strip() or 0),
                    'Education': 1 if entries['Education'].get().strip().lower() == 'graduate' else 0,
                    'Self_Employed': 1 if entries['Self_Employed'].get().strip().lower() in ['yes', '1'] else 0,
                    'ApplicantIncome': float(entries['ApplicantIncome'].strip() or 0),
                    'CoapplicantIncome': float(entries['CoapplicantIncome'].get().strip() or 0),
                    'LoanAmount': float(entries['LoanAmount'].strip() or 0),
                    'Loan_Amount_Term': int(entries['Loan_Amount_Term'].strip() or 360),
                    'Credit_History': float(entries['Credit_History'].strip() or 1.0),
                    'property_Area': 0 if entries['property_Area'].get().strip().lower() == 'urban'
                    else (1 if entries['property_Area'].get().strip().lower() == 'semiurban' else 2)
                }


        input_df = pd.DataFrame([input_data])
        output_df = lem.predict_loan_eligibility(classification_model, input_df.copy())
        
        eligibility = output_df.iloc[0,0]
        max_amount = output_df.iloc[0,1]

        st.subheader("Classification Result")
        if eligibility == "Not Eligible":
              st.write(f"{eligibility}. Maximum Loan Amount: {max_amount:.2f}")
        else:
            st.write(f"{eligibility}")
            
        st.subheader("Loan Recommendation")
        max_loan_amount_pred = lem.predict_max_loan_amount(model, input_df)
        min_loan_duration_pred = lem.predict_min_loan_duration(model, input_data)
        st.write(f"Predicted Max Loan Amount: {max_loan_amount_pred.iloc[0,0]:.2f}")
        st.write(f"Predicted Min Loan Duration: {min_loan_duration_pred.iloc[0,0]}")

    except ValueError as e:
            st.error("Please enter valid inputs.")
    except Exception as e:
           st.error(str(e))
