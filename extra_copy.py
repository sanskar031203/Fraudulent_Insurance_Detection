import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os

class MedicalReportFraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'gender_encoded', 'dizziness', 'disorientation', 'low_bp', 
            'severity_score', 'lab_results', 'prior_admissions', 
            'outcome_severity', 'admission_valid'
        ]
    
    def load_data(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
            required_columns = [
                'age', 'gender', 'dizziness', 'disorientation', 'low_bp', 
                'severity_score', 'lab_results', 'prior_admissions', 
                'outcome_severity', 'admission_valid', 'is_fraudulent'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in the CSV: {missing_columns}")
            
            return data
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return None
    
    def prepare_data(self, dataframe):
        try:
            # Create a copy to avoid modifying the original
            df = dataframe.copy() if isinstance(dataframe, pd.DataFrame) else pd.DataFrame([dataframe])
            
            # Encode gender - handle both string and dictionary input
            if 'gender' in df.columns:
                df['gender_encoded'] = (df['gender'].str.lower() == 'male').astype(int)
            
            X = df[self.feature_columns]
            y = df['is_fraudulent'] if 'is_fraudulent' in df else None
            
            return X, y
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            return None, None
    
    def train_model(self, csv_path):
        data = self.load_data(csv_path)
        if data is None:
            print("Failed to load training data.")
            return
        
        X, y = self.prepare_data(data)
        if X is None or y is None:
            print("Failed to prepare training data.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/fraud_detection_model.joblib')
        joblib.dump(self.scaler, 'models/fraud_detection_scaler.joblib')
    
    def load_model(self):
        try:
            self.model = joblib.load('models/fraud_detection_model.joblib')
            self.scaler = joblib.load('models/fraud_detection_scaler.joblib')
            return True
        except FileNotFoundError:
            print("No pre-trained model found. You need to train the model first.")
            return False
    
    def predict_fraud(self, medical_report_data):
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            # Prepare input data
            X, _ = self.prepare_data(medical_report_data)
            if X is None:
                return None
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            fraud_prediction = self.model.predict(X_scaled)
            fraud_probability = self.model.predict_proba(X_scaled)
            
            return {
                'is_fraudulent': int(fraud_prediction[0]),
                'fraud_probability': float(fraud_probability[0][1])
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

def integrate_fraud_detection(medical_report_analysis, fraud_detector):
    try:
        # Extract data from the input JSON if needed
        data = medical_report_analysis.get('data', medical_report_analysis)
        
        # Predict fraud
        fraud_results = fraud_detector.predict_fraud(data)
        
        # Merge results
        if fraud_results:
            if isinstance(medical_report_analysis, dict):
                medical_report_analysis.update(fraud_results)
            else:
                medical_report_analysis = fraud_results
        
        return medical_report_analysis
    except Exception as e:
        print(f"Error in fraud detection integration: {str(e)}")
        return medical_report_analysis

def datat(data_json):
    try:
        # Initialize fraud detector
        fraud_detector = MedicalReportFraudDetector()
        
        # Path to your CSV file 
        csv_path = r'False_claim_tested_data.csv'
        
        # Train the model using the CSV file
        fraud_detector.train_model(csv_path)
        
        # Ensure data_json is a dictionary with the required fields
        if isinstance(data_json, str):
            data_json = json.loads(data_json)
        
        # Extract data if it's nested
        report_data = data_json.get('data', data_json)
        
        # Integrate fraud detection
        analyzed_report = integrate_fraud_detection(report_data, fraud_detector)
        
        if analyzed_report:
            print("\nMedical Report Analysis with Fraud Detection:")
            print(json.dumps(analyzed_report, indent=2))
        
        return analyzed_report
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        return None