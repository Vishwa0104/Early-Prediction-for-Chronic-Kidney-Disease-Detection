from app import db
from datetime import datetime
import json

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=True)
    age = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Medical test results
    blood_pressure = db.Column(db.Float, nullable=True)
    specific_gravity = db.Column(db.Float, nullable=True)
    albumin = db.Column(db.Float, nullable=True)
    sugar = db.Column(db.Float, nullable=True)
    serum_creatinine = db.Column(db.Float, nullable=True)
    sodium = db.Column(db.Float, nullable=True)
    potassium = db.Column(db.Float, nullable=True)
    calcium = db.Column(db.Float, nullable=True)
    hemoglobin = db.Column(db.Float, nullable=True)
    pcv = db.Column(db.Float, nullable=True)
    gfr = db.Column(db.Float, nullable=True)
    wbc_count = db.Column(db.Float, nullable=True)
    rbc_count = db.Column(db.Float, nullable=True)
    
    # Prediction results
    ckd_risk = db.Column(db.Float, nullable=True)
    severity_level = db.Column(db.String(20), nullable=True)
    survival_probability = db.Column(db.Float, nullable=True)
    prediction_confidence = db.Column(db.Float, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'blood_pressure': self.blood_pressure,
            'specific_gravity': self.specific_gravity,
            'albumin': self.albumin,
            'sugar': self.sugar,
            'serum_creatinine': self.serum_creatinine,
            'sodium': self.sodium,
            'potassium': self.potassium,
            'calcium': self.calcium,
            'hemoglobin': self.hemoglobin,
            'pcv': self.pcv,
            'gfr': self.gfr,
            'wbc_count': self.wbc_count,
            'rbc_count': self.rbc_count,
            'ckd_risk': self.ckd_risk,
            'severity_level': self.severity_level,
            'survival_probability': self.survival_probability,
            'prediction_confidence': self.prediction_confidence,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    prediction_result = db.Column(db.Text, nullable=False)  # JSON string
    feature_importance = db.Column(db.Text, nullable=True)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    patient = db.relationship('Patient', backref=db.backref('predictions', lazy=True))
    
    def get_prediction_result(self):
        return json.loads(self.prediction_result) if self.prediction_result else {}
    
    def set_prediction_result(self, data):
        self.prediction_result = json.dumps(data)
    
    def get_feature_importance(self):
        return json.loads(self.feature_importance) if self.feature_importance else {}
    
    def set_feature_importance(self, data):
        self.feature_importance = json.dumps(data)
