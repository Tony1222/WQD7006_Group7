import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics 
import accuracy_score, confusion_matrix, classification_report

# Specify the correct path to the model file
model_path = 'random_forest_model.pkl'  # Ensure this path is correct

# Path to the local image file
image_path = 'BTC_Img.jpg'  # Use relative path

# Function to read CSV with different encodings
def read_csv_with_encodings(file):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'windows-1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            st.error("The file is empty.")
            return None
    st.error("Failed to read the file with tried encodings.")
    return None

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load the trained model from the specified path
    model = joblib.load(model_path)

    # Title
    st.title("Bitcoin Transaction Fraud Detection")

    # Display an image from the Downloads folder
    if os.path.exists(image_path):
        st.image(image_path, caption='BTC Transaction', use_column_width=True)
    else:
        st.error(f"Image file not found: {image_path}")

    # Upload CSV data for predictions
    uploaded_file = st.file_uploader("Choose a file for predictions")

    if uploaded_file is not None:
        data = read_csv_with_encodings(uploaded_file)
        if data is not None and not data.empty:
            st.write("Data uploaded successfully!")

            # Display the number of transactions
            num_transactions = data.shape[0]
            st.write(f"Number of transactions in the uploaded file: {num_transactions}")

            # Show the first 5 examples
            st.write("Here are the first 5 examples from the uploaded data:")
            st.write(data.head())

            # Check if required columns are present
            if 'address' in data.columns and 'label' in data.columns and 'label_encoded' in data.columns:
                data_to_predict = data.drop(['address', 'label'], axis=1)

                true_labels = data['label_encoded']
                data_to_predict = data_to_predict.drop('label_encoded', axis=1)

                predictions = model.predict(data_to_predict)
                
                # Add predictions to the dataframe
                data['Predicted_label_encoded'] = predictions
                
                # Compare actual label with predicted label_encoded
                data['Correct_Prediction'] = data.apply(
                    lambda row: row['label_encoded'] == row['Predicted_label_encoded'], axis=1)
                
                correct_predictions = data['Correct_Prediction'].sum()
                incorrect_predictions = num_transactions - correct_predictions

                # Calculate prediction accuracy
                accuracy = correct_predictions / num_transactions

                st.write(f"Prediction Accuracy: {accuracy:.2f}")

                # Display correct and incorrect predictions
                st.write(f"Number of correct predictions: {correct_predictions}")
                st.write(f"Number of incorrect predictions: {incorrect_predictions}")

                st.write("Confusion Matrix:")
                st.write(confusion_matrix(true_labels, predictions))

                st.write("Classification Report:")
                st.write(classification_report(true_labels, predictions))

                st.write("Predictions:")
                st.write(data.head(10))  # Show the first 10 rows with predictions
            else:
                st.error("Required columns 'address', 'label', or 'label_encoded' are missing from the uploaded file.")
        else:
            st.error("The uploaded file is empty or improperly formatted.")
