import tkinter as tk
from tkinter import messagebox
import loan_eligibility_model as lem
import pandas as pd
import os


def run_app():
    root = tk.Tk()
    root.title("Loan Eligibility Checker")

    fields = ['Gender', 'Married', 'Dependents', 'Education',
              'Self_Employed', 'ApplicantIncome',
              'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term',
              'Credit_History', 'property_Area']
    entries = {}

    for field in fields:
        label = tk.Label(root, text=field)
        label.pack()
        entry = tk.Entry(root)
        entry.pack()
        entries[field] = entry

    def check_eligibility():
        print("Starting check_eligibility")  # Log
        try:
            # Load data and train the model
            training_data_path = 'training_set.csv'
            training_data = lem.load_data(training_data_path)
            if training_data is None:
                 messagebox.showerror("Error", "Unable to load training data.")
                 print("Unable to load training data.") #Log
                 return

            model, _, _ = lem.train_classification_model(training_data.copy())
            if model is None:
                messagebox.showerror("Error", "Classification model training failed.")
                print("Classification model training failed.") #Log
                return

            # Retrieve and Validate User Inputs
            input_data = {}
            for field in fields:
                value = entries[field].get().strip()
                print(f"Retrieved '{field}': '{value}'")  # Log

                if not value and field not in ['Dependents', 'Loan_Amount_Term', 'Credit_History','LoanAmount']:
                    messagebox.showerror("Input Error", f"Please enter a value for '{field}'")
                    return
                try:
                   if field in ['Dependents', 'Loan_Amount_Term']:
                         input_data[field] = int(value or 0)
                   elif field in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History']:
                         input_data[field] = float(value or 0.0)
                   elif field == 'Gender':
                         input_data[field] = 1 if value.lower() in ['male', '1'] else 0
                   elif field == 'Married':
                         input_data[field] = 1 if value.lower() in ['yes','1'] else 0
                   elif field == 'Education':
                        input_data[field] = 1 if value.lower() == 'graduate' else 0
                   elif field == 'Self_Employed':
                         input_data[field] = 1 if value.lower() in ['yes','1'] else 0
                   elif field == 'property_Area':
                         input_data[field] = 0 if value.lower() == 'urban' else (1 if value.lower() == 'semiurban' else 2)
                   else:
                        input_data[field] = value


                except ValueError:
                     messagebox.showerror("Input Error", f"Invalid input format for '{field}'")
                     return

            print(f"Input data: {input_data}")  # Log
            eligibility, max_amount = lem.predict_loan_eligibility(model, input_data)
            if eligibility is None:
                  messagebox.showerror("Error","Not eligible")
                  print("Not eligible")#Log
                  return
            if eligibility == "Not Eligible":
                messagebox.showinfo("Result", f"{eligibility}. Maximum Loan Amount: {max_amount:.2f}")
            else:
                messagebox.showinfo("Result", eligibility)
            print("check_eligibility function execution complete") #Log
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid inputs.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def check_loan_recommendation():
        print("Starting check_loan_recommendation") #Log
        try:
             # Load data and train the model
            training_data_path = 'training_set.csv'
            training_data = lem.load_data(training_data_path)
            if training_data is None:
                 messagebox.showerror("Error", "Unable to load training data.")
                 print("Unable to load training data")#Log
                 return
            model,_, _ = lem.train_regression_model(training_data.copy())
            if model is None:
                  messagebox.showerror("Error","Regression model training failed.")
                  print("Regression model training failed")#Log
                  return


             # Retrieve and Validate User Inputs
            input_data = {}
            for field in fields:
                value = entries[field].get().strip()
                print(f"Retrieved '{field}': '{value}'") #Log
                if not value and field not in ['Dependents','Loan_Amount_Term','Credit_History','LoanAmount']:
                     messagebox.showerror("Input Error", f"Please enter a value for '{field}'")
                     return
                try:
                    if field in ['Dependents', 'Loan_Amount_Term']:
                        input_data[field] = int(value or 0)
                    elif field in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History']:
                        input_data[field] = float(value or 0.0)
                    elif field == 'Gender':
                         input_data[field] = 1 if value.lower() in ['male', '1'] else 0
                    elif field == 'Married':
                         input_data[field] = 1 if value.lower() in ['yes','1'] else 0
                    elif field == 'Education':
                        input_data[field] = 1 if value.lower() == 'graduate' else 0
                    elif field == 'Self_Employed':
                         input_data[field] = 1 if value.lower() in ['yes','1'] else 0
                    elif field == 'property_Area':
                         input_data[field] = 0 if value.lower() == 'urban' else (1 if value.lower() == 'semiurban' else 2)
                    else:
                        input_data[field] = value
                except ValueError:
                     messagebox.showerror("Input Error", f"Invalid input format for '{field}'")
                     return

            print(f"Input data: {input_data}")#Log
            max_loan_amount = lem.predict_max_loan_amount(model, input_data)
            if max_loan_amount is None:
                 messagebox.showerror("Error", "Loan amount prediction failed")
                 print("Loan amount prediction failed")#Log
                 return
            min_loan_duration = lem.predict_min_loan_duration(model, input_data)
            if min_loan_duration is None:
                messagebox.showerror("Error", "Loan duration prediction failed")
                print("Loan duration prediction failed")#Log
                return

            messagebox.showinfo("Loan Recommendation", f"Predicted Max Loan Amount: {max_loan_amount:.2f}\n"
                 f"Predicted Min Loan Duration: {min_loan_duration if not isinstance(min_loan_duration, str) else min_loan_duration}")
            print("check_loan_recommendation function execution complete")#Log

        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid inputs.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    button_check_eligibility = tk.Button(root, text="Check Eligibility", command=check_eligibility)
    button_check_eligibility.pack()

    button_recommendation = tk.Button(root, text="Get Loan Recommendation", command=check_loan_recommendation)
    button_recommendation.pack()

    root.mainloop()


if __name__ == "__main__":
    run_app()