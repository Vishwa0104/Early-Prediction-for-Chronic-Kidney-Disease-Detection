import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-for-ckd-detection")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///ckd_detection.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

# Register template filters
from utils import get_risk_level, get_severity_badge_class, format_medical_value

@app.template_filter('get_risk_level')
def get_risk_level_filter(risk_score):
    return get_risk_level(risk_score)

@app.template_filter('get_severity_badge_class')
def get_severity_badge_class_filter(severity):
    return get_severity_badge_class(severity)

@app.template_filter('format_medical_value')
def format_medical_value_filter(value, unit='', decimal_places=2):
    return format_medical_value(value, unit, decimal_places)

# Import and register routes
from routes import *
