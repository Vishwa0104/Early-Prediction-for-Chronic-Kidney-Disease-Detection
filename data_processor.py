import pandas as pd
import numpy as np
import logging
from io import StringIO

logger = logging.getLogger(__name__)

class CKDDataProcessor:
    def __init__(self):
        self.required_columns = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
            'serum_creatinine', 'sodium', 'potassium', 'calcium', 'hemoglobin',
            'pcv', 'gfr', 'wbc_count', 'rbc_count'
        ]
        
        self.column_ranges = {
            'age': (18, 90),
            'blood_pressure': (80, 250),
            'specific_gravity': (1.000, 1.040),
            'albumin': (0, 10),
            'sugar': (0, 10),
            'serum_creatinine': (0.1, 20),
            'sodium': (100, 180),
            'potassium': (1.0, 8.0),
            'calcium': (5.0, 15.0),
            'hemoglobin': (3.0, 20.0),
            'pcv': (10, 60),
            'gfr': (1, 200),
            'wbc_count': (1000, 50000),
            'rbc_count': (1.0, 8.0)
        }
    
    def validate_csv_file(self, file_content):
        """Validate uploaded CSV file"""
        try:
            # Try to read CSV
            df = pd.read_csv(StringIO(file_content))
            
            if df.empty:
                return False, "CSV file is empty"
            
            # Check for required columns (allow flexibility in naming)
            df_columns_lower = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            missing_columns = []
            
            for req_col in self.required_columns:
                if req_col not in df_columns_lower:
                    # Try common variations
                    variations = self._get_column_variations(req_col)
                    if not any(var in df_columns_lower for var in variations):
                        missing_columns.append(req_col)
            
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            return True, "CSV file is valid"
            
        except Exception as e:
            return False, f"Error reading CSV file: {str(e)}"
    
    def _get_column_variations(self, column_name):
        """Get common variations of column names"""
        variations = {
            'blood_pressure': ['bp', 'bloodpressure', 'blood_pressure', 'systolic'],
            'specific_gravity': ['sg', 'spec_gravity', 'specific_gravity'],
            'serum_creatinine': ['creatinine', 'serum_creat', 'creat'],
            'wbc_count': ['wbc', 'white_blood_cell', 'wcc'],
            'rbc_count': ['rbc', 'red_blood_cell', 'rcc'],
            'pcv': ['packed_cell_volume', 'hematocrit', 'hct']
        }
        return variations.get(column_name, [column_name])
    
    def process_csv_data(self, file_content):
        """Process and clean CSV data"""
        try:
            df = pd.read_csv(StringIO(file_content))
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Map column variations to standard names
            column_mapping = {}
            for req_col in self.required_columns:
                for col in df.columns:
                    if col == req_col or col in self._get_column_variations(req_col):
                        column_mapping[col] = req_col
                        break
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Select only required columns that exist
            available_cols = [col for col in self.required_columns if col in df.columns]
            df = df[available_cols + [col for col in df.columns if col not in self.required_columns]]
            
            # Clean and validate data
            df = self._clean_data(df)
            
            return df, None
            
        except Exception as e:
            logger.error(f"Error processing CSV data: {str(e)}")
            return None, f"Error processing CSV data: {str(e)}"
    
    def _clean_data(self, df):
        """Clean and validate data values"""
        for column in self.required_columns:
            if column in df.columns:
                # Convert to numeric, handling errors
                df[column] = pd.to_numeric(df[column], errors='coerce')
                
                # Validate ranges
                if column in self.column_ranges:
                    min_val, max_val = self.column_ranges[column]
                    df[column] = df[column].clip(lower=min_val, upper=max_val)
        
        return df
    
    def validate_patient_data(self, patient_data):
        """Validate individual patient data"""
        errors = []
        warnings = []
        
        for field, value in patient_data.items():
            if field in self.required_columns:
                try:
                    numeric_value = float(value) if value is not None else None
                    
                    if numeric_value is not None and field in self.column_ranges:
                        min_val, max_val = self.column_ranges[field]
                        if numeric_value < min_val or numeric_value > max_val:
                            warnings.append(f"{field} value {numeric_value} is outside normal range ({min_val}-{max_val})")
                        
                except ValueError:
                    errors.append(f"Invalid numeric value for {field}: {value}")
        
        return errors, warnings
    
    def prepare_patient_data(self, form_data):
        """Prepare patient data from form input"""
        patient_data = {}
        
        for field in self.required_columns:
            value = form_data.get(field)
            if value and value.strip():
                try:
                    patient_data[field] = float(value)
                except ValueError:
                    logger.warning(f"Could not convert {field} value '{value}' to float")
                    patient_data[field] = None
            else:
                patient_data[field] = None
        
        # Add patient name if provided
        if form_data.get('patient_name'):
            patient_data['name'] = form_data['patient_name'].strip()
        
        return patient_data
    
    def get_data_summary(self, df):
        """Get summary statistics for the dataset"""
        summary = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'basic_stats': {}
        }
        
        for column in self.required_columns:
            if column in df.columns:
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    summary['basic_stats'][column] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'count': int(len(col_data))
                    }
        
        return summary

# Global processor instance
data_processor = CKDDataProcessor()
