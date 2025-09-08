# Overview

This is a Flask-based web application for Chronic Kidney Disease (CKD) detection and monitoring. The system provides AI-powered early detection capabilities through machine learning models that analyze patient medical data. It supports both single patient predictions and batch processing via CSV file uploads, offering a comprehensive dashboard for healthcare professionals to monitor patient risk levels and track medical parameters over time.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses a traditional server-rendered architecture with Flask templates and Bootstrap for responsive UI. The frontend includes multiple pages (dashboard, prediction forms, batch upload, results display) with custom CSS for medical-themed styling and JavaScript for client-side validation and interactivity. Chart.js is integrated for data visualization, and Feather icons provide consistent iconography throughout the interface.

## Backend Architecture
Built on Flask with a modular design pattern separating concerns into distinct components:
- **App Configuration**: Central Flask application setup with database configuration and middleware
- **Database Models**: SQLAlchemy ORM models for patient data and prediction history
- **Route Handlers**: Organized route definitions for different application features
- **ML Processing**: Dedicated machine learning pipeline for CKD risk prediction
- **Data Processing**: CSV validation and medical parameter processing utilities

## Data Storage
Uses SQLAlchemy with flexible database backend support (SQLite for development, configurable for production databases like PostgreSQL). The database schema stores comprehensive patient medical data including lab results, vital signs, and prediction outcomes. Models include proper relationships and indexing for efficient querying.

## Machine Learning Pipeline
Implements multiple ML algorithms (Random Forest, Logistic Regression, XGBoost) for CKD risk assessment. The system includes:
- Synthetic training data generation for development/testing
- Feature scaling and preprocessing pipelines
- Model training and validation workflows
- Risk scoring and severity classification
- Confidence metrics for prediction reliability

## Authentication and Authorization
Currently implements basic session management through Flask's built-in session handling. The application uses environment-based secret key configuration for session security. No complex user authentication system is implemented, suggesting single-user or trusted environment deployment.

# External Dependencies

## Core Framework Dependencies
- **Flask**: Primary web framework with SQLAlchemy extension for database ORM
- **Werkzeug**: WSGI utilities including ProxyFix middleware for reverse proxy compatibility

## Machine Learning Libraries
- **scikit-learn**: Core ML algorithms and preprocessing utilities
- **XGBoost**: Gradient boosting framework for advanced predictions
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computing foundation

## Frontend Libraries
- **Bootstrap**: UI framework with dark theme customization
- **Chart.js**: Client-side charting and data visualization
- **Feather Icons**: Scalable vector icon library

## Data Processing
- **pandas**: CSV file parsing and data validation
- **StringIO**: In-memory file processing for uploaded content

## Database Support
- **SQLite**: Default development database (via SQLAlchemy)
- **Configurable**: Environment-based database URL support for production databases