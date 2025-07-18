#!/usr/bin/env python3
"""
Quadratic Neural Network Web Application Configuration
Flask Application Configuration Settings

Author: Matt
Location: Varna, Bulgaria
Date: July 2025

Configuration settings for the beautiful Apple-like web interface
"""

import os
import sys
from pathlib import Path
from datetime import timedelta

# Add the Neural Engine path for imports
current_dir = Path(__file__).parent
neural_engine_root = current_dir.parent.parent  # Go up two levels
sys.path.insert(0, str(neural_engine_root))

class Config:
    """Base configuration class"""
    
    # Basic Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'quadratic-neural-network-varna-2025-production'
    
    # Application Information
    APP_NAME = "Quadratic Neural Network"
    APP_VERSION = "2.0.0"
    APP_AUTHOR = "Matt"
    APP_LOCATION = "Varna, Bulgaria"
    APP_DESCRIPTION = "Advanced neural network analysis for quadratic equations"
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    # Database Configuration (for future expansion)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS Configuration
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization']
    CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    
    # Neural Network Configuration
    DEFAULT_EPOCHS = 1000
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_VALIDATION_SPLIT = 0.15
    DEFAULT_TEST_SPLIT = 0.15
    
    # Training Configuration
    MAX_TRAINING_TIME = 3600  # 1 hour max training time
    TRAINING_TIMEOUT = 300    # 5 minutes timeout
    MAX_CONCURRENT_TRAININGS = 1
    
    # Model Configuration
    MODEL_SAVE_PATH = 'models'
    MODEL_BACKUP_PATH = 'models/backups'
    MAX_MODEL_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Data Processing Configuration
    MAX_DATASET_SIZE = 1000000  # 1M equations max
    SAMPLE_SIZE_PREVIEW = 100   # Preview first 100 rows
    DATA_VALIDATION_STRICT = True
    
    # Performance Configuration
    CONFIDENCE_ESTIMATION_SAMPLES = 50
    PREDICTION_BATCH_SIZE = 1000
    ANALYSIS_TIMEOUT = 60  # 1 minute
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Security Configuration
    WTF_CSRF_TIME_LIMIT = 3600
    WTF_CSRF_SSL_STRICT = True
    
    # Cache Configuration
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_DEFAULT = "100 per hour"
    
    # Neural Engine Specific Configuration
    NEURAL_ENGINE_PATH = str(parent_dir)
    USE_CUSTOM_NEURAL_ENGINE = True
    
    # Default Network Architectures
    DEFAULT_ARCHITECTURES = {
        'coeff_to_roots': [3, 16, 32, 16, 2],
        'partial_coeff_to_missing': [3, 20, 24, 12, 2],
        'roots_to_coeff': [2, 20, 32, 20, 3],
        'single_missing': [4, 24, 32, 16, 1],
        'verify_equation': [5, 32, 24, 16, 1]
    }
    
    # Default Activation Functions
    DEFAULT_ACTIVATIONS = {
        'coeff_to_roots': ['relu', 'relu', 'relu', 'linear'],
        'partial_coeff_to_missing': ['relu', 'swish', 'relu', 'linear'],
        'roots_to_coeff': ['relu', 'swish', 'relu', 'linear'],
        'single_missing': ['relu', 'swish', 'relu', 'linear'],
        'verify_equation': ['relu', 'swish', 'relu', 'sigmoid']
    }
    
    # Scenario Colors
    SCENARIO_COLORS = {
        'coeff_to_roots': '#FF6B6B',
        'partial_coeff_to_missing': '#4ECDC4',
        'roots_to_coeff': '#45B7D1',
        'single_missing': '#96CEB4',
        'verify_equation': '#FFEAA7'
    }
    
    # API Configuration
    API_TITLE = "Quadratic Neural Network API"
    API_VERSION = "v1"
    API_DESCRIPTION = "RESTful API for quadratic neural network analysis"
    
    # Export Configuration
    EXPORT_FORMATS = ['json', 'csv', 'txt', 'pdf']
    EXPORT_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Monitoring Configuration
    ENABLE_METRICS = True
    METRICS_PORT = 9090
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration"""
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_BACKUP_PATH, exist_ok=True)
        
        # Set up logging
        import logging
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=Config.LOG_FORMAT
        )
        
        # Validate Neural Engine availability
        try:
            from nn_core import NeuralNetwork
            from autodiff import TrainingEngine, Adam
            app.logger.info("✅ Neural Engine components available")
        except ImportError as e:
            app.logger.error(f"❌ Neural Engine not available: {e}")
            raise RuntimeError("Neural Engine required for application")


class DevelopmentConfig(Config):
    """Development configuration"""
    
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    SECRET_KEY = 'development-key-not-for-production'
    
    # Relaxed security for development
    SESSION_COOKIE_SECURE = False
    WTF_CSRF_SSL_STRICT = False
    
    # More verbose logging
    LOG_LEVEL = 'DEBUG'
    
    # Faster training for development
    DEFAULT_EPOCHS = 100
    
    # Allow more file types in development
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'json', 'xlsx'}
    
    # Development database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev_quadratic.db'
    
    # Disable rate limiting in development
    RATELIMIT_ENABLED = False
    
    @staticmethod
    def init_app(app):
        """Initialize development configuration"""
        Config.init_app(app)
        app.logger.info("🚧 Running in DEVELOPMENT mode")
        
        # Enable Flask debug toolbar if available
        try:
            from flask_debugtoolbar import DebugToolbarExtension
            app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
            toolbar = DebugToolbarExtension(app)
        except ImportError:
            pass


class ProductionConfig(Config):
    """Production configuration"""
    
    DEBUG = False
    TESTING = False
    
    # Production security settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or Config.SECRET_KEY
    
    # Strict security in production
    SESSION_COOKIE_SECURE = True
    WTF_CSRF_SSL_STRICT = True
    
    # Production database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///prod_quadratic.db'
    
    # Error handling
    PROPAGATE_EXCEPTIONS = False
    
    # Rate limiting enabled
    RATELIMIT_ENABLED = True
    
    # Reduced file size limits
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    @staticmethod
    def init_app(app):
        """Initialize production configuration"""
        Config.init_app(app)
        app.logger.info("🚀 Running in PRODUCTION mode")
        
        # Set up production logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        # File handler for errors
        file_handler = RotatingFileHandler(
            'logs/quadratic_neural_network.log',
            maxBytes=Config.LOG_MAX_BYTES,
            backupCount=Config.LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        file_handler.setLevel(logging.WARNING)
        app.logger.addHandler(file_handler)
        
        # Email handler for critical errors (if configured)
        if os.environ.get('MAIL_SERVER'):
            from logging.handlers import SMTPHandler
            mail_handler = SMTPHandler(
                mailhost=os.environ.get('MAIL_SERVER'),
                fromaddr=os.environ.get('MAIL_FROM'),
                toaddrs=os.environ.get('MAIL_TO').split(','),
                subject='Quadratic Neural Network Error'
            )
            mail_handler.setLevel(logging.ERROR)
            app.logger.addHandler(mail_handler)


class TestingConfig(Config):
    """Testing configuration"""
    
    TESTING = True
    DEBUG = True
    
    # Testing-specific settings
    SECRET_KEY = 'testing-key'
    WTF_CSRF_ENABLED = False
    
    # In-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Faster training for tests
    DEFAULT_EPOCHS = 10
    
    # Small file limits for testing
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB
    
    # Disable rate limiting in tests
    RATELIMIT_ENABLED = False
    
    @staticmethod
    def init_app(app):
        """Initialize testing configuration"""
        Config.init_app(app)
        app.logger.info("🧪 Running in TESTING mode")


class DockerConfig(ProductionConfig):
    """Docker container configuration"""
    
    # Docker-specific settings
    HOST = '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Use environment variables for configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://user:password@db:5432/quadratic'
    
    # Redis for caching in Docker
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL') or 'redis://redis:6379/0'
    
    # Docker volume paths
    UPLOAD_FOLDER = '/app/uploads'
    MODEL_SAVE_PATH = '/app/models'
    
    @staticmethod
    def init_app(app):
        """Initialize Docker configuration"""
        ProductionConfig.init_app(app)
        app.logger.info("🐳 Running in DOCKER mode")


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    return config[os.environ.get('FLASK_ENV', 'default')]

# Neural Engine Integration Settings
NEURAL_ENGINE_SETTINGS = {
    'use_custom_engine': True,
    'engine_path': str(parent_dir),
    'required_modules': [
        'nn_core',
        'autodiff',
        'data_utils'
    ],
    'default_optimizer': 'Adam',
    'default_loss_function': 'mean_squared_error',
    'activation_functions': [
        'relu',
        'swish',
        'linear',
        'sigmoid',
        'tanh'
    ]
}

# Application Constants
APP_CONSTANTS = {
    'MAX_EQUATION_COEFFICIENTS': 5,
    'QUADRATIC_EQUATION_FEATURES': ['a', 'b', 'c', 'x1', 'x2'],
    'PERFORMANCE_THRESHOLDS': {
        'excellent': 0.9,
        'good': 0.7,
        'fair': 0.5,
        'poor': 0.0
    },
    'CONFIDENCE_THRESHOLDS': {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.0
    },
    'DEFAULT_PRECISION': 6,
    'EQUATION_TOLERANCE': 1e-6
}

# Validation Rules
VALIDATION_RULES = {
    'coefficients': {
        'min_value': -1000,
        'max_value': 1000,
        'precision': 10
    },
    'roots': {
        'min_value': -1000,
        'max_value': 1000,
        'precision': 10
    },
    'dataset': {
        'min_rows': 100,
        'max_rows': 1000000,
        'required_columns': ['a', 'b', 'c', 'x1', 'x2']
    }
}

# UI Configuration
UI_CONFIG = {
    'theme': 'apple-like',
    'primary_color': '#007AFF',
    'secondary_color': '#5856D6',
    'success_color': '#34C759',
    'warning_color': '#FF9500',
    'error_color': '#FF3B30',
    'chart_colors': [
        '#FF6B6B', '#4ECDC4', '#45B7D1', 
        '#96CEB4', '#FFEAA7', '#D63031'
    ],
    'animation_duration': 300,
    'notification_timeout': 5000
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    'enable_monitoring': True,
    'metrics_collection': True,
    'performance_alerts': True,
    'slow_query_threshold': 1.0,
    'memory_usage_threshold': 0.8,
    'cpu_usage_threshold': 0.9
}

# Feature Flags
FEATURE_FLAGS = {
    'enable_batch_prediction': True,
    'enable_model_comparison': True,
    'enable_advanced_analysis': True,
    'enable_export_functionality': True,
    'enable_real_time_training': True,
    'enable_model_versioning': False,
    'enable_user_authentication': False,
    'enable_api_documentation': True
}

# Version Information
VERSION_INFO = {
    'version': '2.0.0',
    'release_date': '2025-07-18',
    'author': 'Matt',
    'location': 'Varna, Bulgaria',
    'description': 'Advanced neural network analysis for quadratic equations',
    'neural_engine_version': '1.0.0',
    'flask_version': '2.3.3'
}

if __name__ == '__main__':
    # Configuration validation
    config_class = get_config()
    print(f"🔧 Configuration: {config_class.__name__}")
    print(f"📍 Location: {Config.APP_LOCATION}")
    print(f"🧠 Neural Engine: {NEURAL_ENGINE_SETTINGS['use_custom_engine']}")
    print(f"✅ Configuration validated successfully!")
