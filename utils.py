import os
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_risk_level(risk_score):
    """Convert risk score to human-readable level"""
    if risk_score >= 0.7:
        return 'High', 'danger'
    elif risk_score >= 0.3:
        return 'Moderate', 'warning'
    else:
        return 'Low', 'success'

def format_medical_value(value, unit='', decimal_places=2):
    """Format medical values for display"""
    if value is None:
        return 'N/A'
    try:
        formatted_value = f"{float(value):.{decimal_places}f}"
        return f"{formatted_value} {unit}".strip()
    except (ValueError, TypeError):
        return str(value)

def get_severity_badge_class(severity):
    """Get Bootstrap badge class for severity level"""
    severity_classes = {
        'Mild': 'bg-success',
        'Moderate': 'bg-warning',
        'Severe': 'bg-danger'
    }
    return severity_classes.get(severity, 'bg-secondary')

def calculate_age_group(age):
    """Calculate age group for analytics"""
    if age < 30:
        return '< 30'
    elif age < 50:
        return '30-49'
    elif age < 70:
        return '50-69'
    else:
        return '70+'

def get_normal_ranges():
    """Get normal ranges for medical parameters"""
    return {
        'serum_creatinine': {'min': 0.6, 'max': 1.3, 'unit': 'mg/dL'},
        'gfr': {'min': 90, 'max': 150, 'unit': 'mL/min/1.73m²'},
        'hemoglobin': {'min': 12.0, 'max': 16.0, 'unit': 'g/dL'},
        'blood_pressure': {'min': 90, 'max': 140, 'unit': 'mmHg'},
        'sodium': {'min': 135, 'max': 145, 'unit': 'mEq/L'},
        'potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mEq/L'},
        'calcium': {'min': 8.5, 'max': 10.5, 'unit': 'mg/dL'},
        'albumin': {'min': 0, 'max': 1, 'unit': 'g/dL'},
        'sugar': {'min': 0, 'max': 1, 'unit': 'g/dL'},
        'pcv': {'min': 35, 'max': 50, 'unit': '%'},
        'wbc_count': {'min': 4000, 'max': 11000, 'unit': '/µL'},
        'rbc_count': {'min': 4.0, 'max': 6.0, 'unit': 'million/µL'}
    }

def is_value_normal(parameter, value):
    """Check if a medical parameter value is within normal range"""
    ranges = get_normal_ranges()
    if parameter in ranges and value is not None:
        return ranges[parameter]['min'] <= float(value) <= ranges[parameter]['max']
    return None

def get_risk_recommendations(risk_score, severity, patient_data):
    """Get medical recommendations based on risk assessment"""
    recommendations = []
    
    if risk_score > 0.7:
        recommendations.extend([
            "Immediate nephrology consultation recommended",
            "Close monitoring of kidney function required",
            "Consider medication review and adjustment"
        ])
    elif risk_score > 0.3:
        recommendations.extend([
            "Regular follow-up with primary care physician",
            "Monitor blood pressure and diabetes if present",
            "Lifestyle modifications recommended"
        ])
    else:
        recommendations.extend([
            "Continue routine health maintenance",
            "Annual kidney function screening",
            "Maintain healthy lifestyle"
        ])
    
    # Specific parameter-based recommendations
    if patient_data.get('serum_creatinine', 0) > 1.3:
        recommendations.append("Elevated creatinine - avoid nephrotoxic medications")
    
    if patient_data.get('gfr', 100) < 60:
        recommendations.append("Reduced GFR - consider dietary protein restriction")
    
    if patient_data.get('blood_pressure', 120) > 140:
        recommendations.append("Blood pressure management is crucial for kidney health")
    
    if patient_data.get('hemoglobin', 12) < 10:
        recommendations.append("Low hemoglobin may indicate CKD progression")
    
    return recommendations
