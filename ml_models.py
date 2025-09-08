import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class CKDMLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_synthetic_training_data(self):
        """
        Create synthetic training data based on CKD medical parameters
        This would normally be replaced with real medical data
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic medical data with realistic ranges
        data = {
            'age': np.random.normal(55, 15, n_samples).clip(18, 90),
            'blood_pressure': np.random.normal(140, 25, n_samples).clip(90, 200),
            'specific_gravity': np.random.normal(1.020, 0.007, n_samples).clip(1.005, 1.030),
            'albumin': np.random.exponential(1.0, n_samples).clip(0, 5),
            'sugar': np.random.exponential(0.5, n_samples).clip(0, 4),
            'serum_creatinine': np.random.lognormal(0.5, 0.6, n_samples).clip(0.5, 15),
            'sodium': np.random.normal(140, 8, n_samples).clip(120, 160),
            'potassium': np.random.normal(4.2, 0.8, n_samples).clip(2.5, 7.0),
            'calcium': np.random.normal(9.5, 1.0, n_samples).clip(7.0, 12.0),
            'hemoglobin': np.random.normal(12.5, 2.5, n_samples).clip(6.0, 18.0),
            'pcv': np.random.normal(38, 8, n_samples).clip(20, 55),
            'gfr': np.random.normal(70, 30, n_samples).clip(5, 150),
            'wbc_count': np.random.normal(8000, 3000, n_samples).clip(2000, 20000),
            'rbc_count': np.random.normal(4.5, 1.0, n_samples).clip(2.0, 7.0)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic CKD labels based on medical indicators
        ckd_probability = (
            (df['serum_creatinine'] > 1.5).astype(int) * 0.4 +
            (df['gfr'] < 60).astype(int) * 0.3 +
            (df['albumin'] > 1).astype(int) * 0.2 +
            (df['hemoglobin'] < 10).astype(int) * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        ).clip(0, 1)
        
        df['ckd'] = (ckd_probability > 0.5).astype(int)
        
        # Create severity levels
        severity_conditions = [
            (df['gfr'] >= 60) & (df['serum_creatinine'] <= 1.5),  # Mild
            (df['gfr'] >= 30) & (df['gfr'] < 60),  # Moderate
            df['gfr'] < 30  # Severe
        ]
        severity_labels = ['Mild', 'Moderate', 'Severe']
        df['severity'] = np.select(severity_conditions, severity_labels, default='Mild')
        
        # Create survival probability (higher GFR and hemoglobin = better survival)
        df['survival_months'] = (
            df['gfr'] * 0.5 + 
            df['hemoglobin'] * 2 + 
            np.random.normal(50, 15, n_samples)
        ).clip(6, 120)
        
        return df
    
    def train_models(self, df=None):
        """Train all ML models for CKD prediction"""
        if df is None:
            df = self.prepare_synthetic_training_data()
        
        logger.info("Starting model training...")
        
        # Feature columns
        feature_columns = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'serum_creatinine', 'sodium', 'potassium', 'calcium', 'hemoglobin',
            'pcv', 'gfr', 'wbc_count', 'rbc_count'
        ]
        
        self.feature_names = feature_columns
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Train CKD Detection Model (Binary Classification)
        y_ckd = df['ckd']
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_ckd, test_size=0.2, random_state=42, stratify=y_ckd
        )
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(random_state=42, max_iter=1000)
        self.models['logistic_regression'].fit(X_train, y_train)
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['random_forest'].fit(X_train, y_train)
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        self.models['xgboost'].fit(X_train, y_train)
        
        # Train Severity Classification Model
        y_severity = df['severity']
        self.label_encoders['severity'] = LabelEncoder()
        y_severity_encoded = self.label_encoders['severity'].fit_transform(y_severity)
        
        self.models['severity_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['severity_classifier'].fit(X_scaled, y_severity_encoded)
        
        # Train Survival Prediction Model (Regression)
        y_survival = df['survival_months']
        self.models['survival_predictor'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['survival_predictor'].fit(X_scaled, y_survival)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        self.is_trained = True
        logger.info("Model training completed successfully")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            model = self.models[model_name]
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"{model_name} accuracy: {accuracy:.3f}")
    
    def predict_ckd_risk(self, patient_data, model_name='random_forest'):
        """Predict CKD risk for a single patient"""
        if not self.is_trained:
            self.train_models()
            
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df[self.feature_names]
        patient_df = patient_df.fillna(patient_df.median())
        
        # Scale features
        patient_scaled = self.scalers['standard'].transform(patient_df)
        
        # Make prediction
        model = self.models[model_name]
        risk_probability = model.predict_proba(patient_scaled)[0][1]
        prediction = model.predict(patient_scaled)[0]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, model.feature_importances_))
        else:
            # For logistic regression
            importance = dict(zip(self.feature_names, abs(model.coef_[0])))
        
        return {
            'risk_probability': float(risk_probability),
            'has_ckd': bool(prediction),
            'confidence': float(max(model.predict_proba(patient_scaled)[0])),
            'feature_importance': importance
        }
    
    def predict_severity(self, patient_data):
        """Predict CKD severity level"""
        if not self.is_trained:
            self.train_models()
            
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df[self.feature_names]
        patient_df = patient_df.fillna(patient_df.median())
        
        # Scale features
        patient_scaled = self.scalers['standard'].transform(patient_df)
        
        # Make prediction
        model = self.models['severity_classifier']
        severity_encoded = model.predict(patient_scaled)[0]
        severity_proba = model.predict_proba(patient_scaled)[0]
        
        severity_level = self.label_encoders['severity'].inverse_transform([severity_encoded])[0]
        confidence = float(max(severity_proba))
        
        return {
            'severity_level': severity_level,
            'confidence': confidence,
            'probabilities': {
                label: float(prob) for label, prob in 
                zip(self.label_encoders['severity'].classes_, severity_proba)
            }
        }
    
    def predict_survival(self, patient_data):
        """Predict survival probability"""
        if not self.is_trained:
            self.train_models()
            
        # Prepare patient data
        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df[self.feature_names]
        patient_df = patient_df.fillna(patient_df.median())
        
        # Scale features
        patient_scaled = self.scalers['standard'].transform(patient_df)
        
        # Make prediction
        model = self.models['survival_predictor']
        survival_months = model.predict(patient_scaled)[0]
        
        # Convert to probability (higher months = higher probability)
        survival_probability = min(1.0, max(0.1, survival_months / 120.0))
        
        return {
            'survival_months': float(max(0, survival_months)),
            'survival_probability': float(survival_probability)
        }
    
    def get_comprehensive_prediction(self, patient_data):
        """Get all predictions for a patient"""
        ckd_result = self.predict_ckd_risk(patient_data)
        severity_result = self.predict_severity(patient_data)
        survival_result = self.predict_survival(patient_data)
        
        return {
            'ckd_detection': ckd_result,
            'severity_prediction': severity_result,
            'survival_analysis': survival_result,
            'overall_risk_score': (
                ckd_result['risk_probability'] * 0.5 +
                (1 - survival_result['survival_probability']) * 0.3 +
                {'Mild': 0.1, 'Moderate': 0.5, 'Severe': 0.9}.get(severity_result['severity_level'], 0.5) * 0.2
            )
        }

# Global model instance
ml_models = CKDMLModels()
