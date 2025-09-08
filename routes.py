from flask import render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import pandas as pd
import json
import logging
from app import app, db
from models import Patient, PredictionHistory
from ml_models import ml_models
from data_processor import data_processor
from utils import allowed_file, get_risk_level, format_medical_value

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    recent_patients = Patient.query.order_by(Patient.created_at.desc()).limit(5).all()
    total_patients = Patient.query.count()
    high_risk_patients = Patient.query.filter(Patient.ckd_risk > 0.7).count()
    
    return render_template('index.html', 
                         recent_patients=recent_patients,
                         total_patients=total_patients,
                         high_risk_patients=high_risk_patients)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file upload and batch processing"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Read and validate file
                file_content = file.read().decode('utf-8')
                is_valid, message = data_processor.validate_csv_file(file_content)
                
                if not is_valid:
                    flash(f'Invalid CSV file: {message}', 'error')
                    return redirect(request.url)
                
                # Process CSV data
                df, error = data_processor.process_csv_data(file_content)
                if error:
                    flash(f'Error processing file: {error}', 'error')
                    return redirect(request.url)
                
                # Store in session for batch processing
                session['upload_data'] = df.to_json()
                session['upload_filename'] = secure_filename(file.filename)
                
                flash(f'File uploaded successfully! {len(df)} records found.', 'success')
                return redirect(url_for('batch_process'))
                
            except Exception as e:
                logger.error(f"Error uploading file: {str(e)}")
                flash(f'Error uploading file: {str(e)}', 'error')
        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
    
    return render_template('upload.html')

@app.route('/batch_process')
def batch_process():
    """Process uploaded batch data"""
    if 'upload_data' not in session:
        flash('No data to process. Please upload a file first.', 'warning')
        return redirect(url_for('upload'))
    
    try:
        # Load data from session
        df = pd.read_json(session['upload_data'])
        filename = session.get('upload_filename', 'unknown.csv')
        
        processed_results = []
        
        # Process each row
        for index, row in df.iterrows():
            try:
                patient_data = row.to_dict()
                
                # Validate patient data
                errors, warnings = data_processor.validate_patient_data(patient_data)
                
                if not errors:  # Only process if no errors
                    # Get ML predictions
                    predictions = ml_models.get_comprehensive_prediction(patient_data)
                    
                    # Create patient record
                    patient = Patient(
                        name=patient_data.get('name', f'Patient_{index+1}'),
                        age=patient_data.get('age'),
                        blood_pressure=patient_data.get('blood_pressure'),
                        specific_gravity=patient_data.get('specific_gravity'),
                        albumin=patient_data.get('albumin'),
                        sugar=patient_data.get('sugar'),
                        serum_creatinine=patient_data.get('serum_creatinine'),
                        sodium=patient_data.get('sodium'),
                        potassium=patient_data.get('potassium'),
                        calcium=patient_data.get('calcium'),
                        hemoglobin=patient_data.get('hemoglobin'),
                        pcv=patient_data.get('pcv'),
                        gfr=patient_data.get('gfr'),
                        wbc_count=patient_data.get('wbc_count'),
                        rbc_count=patient_data.get('rbc_count'),
                        ckd_risk=predictions['ckd_detection']['risk_probability'],
                        severity_level=predictions['severity_prediction']['severity_level'],
                        survival_probability=predictions['survival_analysis']['survival_probability'],
                        prediction_confidence=predictions['ckd_detection']['confidence']
                    )
                    
                    db.session.add(patient)
                    db.session.flush()  # Get patient ID
                    
                    # Store prediction history
                    history = PredictionHistory(
                        patient_id=patient.id,
                        model_name='comprehensive',
                        prediction_result=json.dumps(predictions),
                        feature_importance=json.dumps(predictions['ckd_detection']['feature_importance'])
                    )
                    db.session.add(history)
                    
                    processed_results.append({
                        'patient_name': patient.name,
                        'ckd_risk': predictions['ckd_detection']['risk_probability'],
                        'severity': predictions['severity_prediction']['severity_level'],
                        'survival_prob': predictions['survival_analysis']['survival_probability'],
                        'status': 'success',
                        'warnings': warnings
                    })
                else:
                    processed_results.append({
                        'patient_name': patient_data.get('name', f'Patient_{index+1}'),
                        'status': 'error',
                        'errors': errors,
                        'warnings': warnings
                    })
                    
            except Exception as e:
                logger.error(f"Error processing patient {index}: {str(e)}")
                processed_results.append({
                    'patient_name': f'Patient_{index+1}',
                    'status': 'error',
                    'errors': [str(e)]
                })
        
        # Commit all changes
        db.session.commit()
        
        # Clear session data
        session.pop('upload_data', None)
        session.pop('upload_filename', None)
        
        flash(f'Batch processing completed! {len([r for r in processed_results if r["status"] == "success"])} patients processed successfully.', 'success')
        
        return render_template('results.html', 
                             results=processed_results,
                             batch_mode=True,
                             filename=filename)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        flash(f'Error processing batch: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Single patient prediction"""
    if request.method == 'POST':
        try:
            # Prepare patient data from form
            patient_data = data_processor.prepare_patient_data(request.form)
            
            # Validate data
            errors, warnings = data_processor.validate_patient_data(patient_data)
            
            if errors:
                for error in errors:
                    flash(error, 'error')
                return render_template('predict.html', patient_data=patient_data)
            
            if warnings:
                for warning in warnings:
                    flash(warning, 'warning')
            
            # Get predictions
            predictions = ml_models.get_comprehensive_prediction(patient_data)
            
            # Save to database
            patient = Patient(
                name=patient_data.get('name', 'Anonymous'),
                age=patient_data.get('age'),
                blood_pressure=patient_data.get('blood_pressure'),
                specific_gravity=patient_data.get('specific_gravity'),
                albumin=patient_data.get('albumin'),
                sugar=patient_data.get('sugar'),
                serum_creatinine=patient_data.get('serum_creatinine'),
                sodium=patient_data.get('sodium'),
                potassium=patient_data.get('potassium'),
                calcium=patient_data.get('calcium'),
                hemoglobin=patient_data.get('hemoglobin'),
                pcv=patient_data.get('pcv'),
                gfr=patient_data.get('gfr'),
                wbc_count=patient_data.get('wbc_count'),
                rbc_count=patient_data.get('rbc_count'),
                ckd_risk=predictions['ckd_detection']['risk_probability'],
                severity_level=predictions['severity_prediction']['severity_level'],
                survival_probability=predictions['survival_analysis']['survival_probability'],
                prediction_confidence=predictions['ckd_detection']['confidence']
            )
            
            db.session.add(patient)
            db.session.flush()
            
            # Store prediction history
            history = PredictionHistory(
                patient_id=patient.id,
                model_name='comprehensive',
                prediction_result=json.dumps(predictions),
                feature_importance=json.dumps(predictions['ckd_detection']['feature_importance'])
            )
            db.session.add(history)
            db.session.commit()
            
            flash('Prediction completed successfully!', 'success')
            return render_template('results.html', 
                                 patient=patient,
                                 predictions=predictions,
                                 batch_mode=False)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            flash(f'Error making prediction: {str(e)}', 'error')
    
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    """Advanced dashboard with analytics"""
    patients = Patient.query.all()
    
    # Calculate statistics
    stats = {
        'total_patients': len(patients),
        'high_risk_count': len([p for p in patients if p.ckd_risk and p.ckd_risk > 0.7]),
        'moderate_risk_count': len([p for p in patients if p.ckd_risk and 0.3 < p.ckd_risk <= 0.7]),
        'low_risk_count': len([p for p in patients if p.ckd_risk and p.ckd_risk <= 0.3]),
        'severity_distribution': {
            'Mild': len([p for p in patients if p.severity_level == 'Mild']),
            'Moderate': len([p for p in patients if p.severity_level == 'Moderate']),
            'Severe': len([p for p in patients if p.severity_level == 'Severe'])
        }
    }
    
    return render_template('dashboard.html', patients=patients, stats=stats)

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    """Detailed view of a specific patient"""
    patient = Patient.query.get_or_404(patient_id)
    predictions = PredictionHistory.query.filter_by(patient_id=patient_id).order_by(PredictionHistory.created_at.desc()).all()
    
    return render_template('patient_detail.html', patient=patient, predictions=predictions)

@app.route('/api/feature_importance/<int:patient_id>')
def get_feature_importance(patient_id):
    """API endpoint for feature importance data"""
    history = PredictionHistory.query.filter_by(patient_id=patient_id).order_by(PredictionHistory.created_at.desc()).first()
    
    if history:
        feature_importance = history.get_feature_importance()
        return jsonify(feature_importance)
    else:
        return jsonify({}), 404

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('upload'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return render_template('500.html'), 500
