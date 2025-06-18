import os
from datetime import timedelta

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Server Configuration
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    
    # Timezone Configuration
    TIMEZONE = os.environ.get('TIMEZONE', 'UTC')
    
    # Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', 'breast_cancer_model.pkl')
    SCALER_PATH = os.environ.get('SCALER_PATH', 'breast_cancer_scaler.pkl')
    
    # Data Processing Configuration
    MAX_FEATURES = 30
    RANDOM_STATE = int(os.environ.get('RANDOM_STATE', 42))
    
    # Machine Learning Configuration
    TEST_SIZE = float(os.environ.get('TEST_SIZE', 0.2))
    
    # Request Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # JSON Configuration
    JSON_SORT_KEYS = True
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # Security Headers
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(hours=12)
    
    # Application Metadata
    APP_NAME = "Breast Cancer Predictor"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Flask web application for breast cancer prediction using machine learning"
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    
    # Feature Names (for documentation/validation)
    FEATURE_NAMES = [f'feature_{i+1}' for i in range(30)]
    
    # Prediction Configuration
    PREDICTION_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.7
    
    # API Configuration
    API_RATE_LIMIT = "100 per hour"
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = '127.0.0.1'
    PORT = 5000

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(cls.LOG_FORMAT)
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SECRET_KEY = 'test-secret-key'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
