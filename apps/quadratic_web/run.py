#!/usr/bin/env python3
"""
Quadratic Neural Network Web Application
Application Entry Point & Startup Script

Production-ready startup script for the beautiful Apple-like web interface
"""

import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime
import argparse
import webbrowser
from threading import Timer

# Add the Neural Engine path
current_dir = Path(__file__).parent
neural_engine_root = current_dir.parent.parent  # Go up two levels
sys.path.insert(0, str(neural_engine_root))

# Import configuration
from config.config import get_config, NEURAL_ENGINE_SETTINGS, VERSION_INFO

def setup_logging(config_class):
    """Setup application logging"""
    log_level = getattr(logging, config_class.LOG_LEVEL.upper())
    
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=config_class.LOG_FORMAT,
        handlers=[
            logging.FileHandler(logs_dir / 'quadratic_app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info("[START] Quadratic Neural Network Web Application")
    logger.info("=" * 60)
    logger.info(f"Version: {VERSION_INFO['version']}")
    logger.info(f"Author: {VERSION_INFO['author']}")
    logger.info(f"Location: {VERSION_INFO['location']}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return logger

def check_neural_engine():
    """Verify Neural Engine availability"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check required modules
        required_modules = NEURAL_ENGINE_SETTINGS['required_modules']
        
        for module_name in required_modules:
            try:
                __import__(module_name)
                logger.info(f"[OK] Neural Engine module '{module_name}' available")
            except ImportError as e:
                logger.error(f"[ERROR] Neural Engine module '{module_name}' not found: {e}")
                return False
        
        # Test basic functionality
        from nn_core import NeuralNetwork
        from autodiff import TrainingEngine, Adam
        
        # Create a simple test network
        test_network = NeuralNetwork(
            [2, 4, 1],
            ['relu', 'linear'],
            device=os.getenv('NEURAL_ENGINE_DEVICE', 'auto')
        )
        test_optimizer = Adam(learning_rate=0.001)
        
        logger.info("[OK] Neural Engine basic functionality verified")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Neural Engine verification failed: {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'flask',
        'flask_cors',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.debug(f"[OK] Package '{package}' available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"[ERROR] Package '{package}' not found")
    
    if missing_packages:
        logger.error("Missing required packages:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        logger.error("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("[OK] All required dependencies available")
    return True

def create_app():
    """Create and configure Flask application"""
    logger = logging.getLogger(__name__)
    
    try:
        # Import Flask app
        from app import app
        
        # Get configuration
        config_class = get_config()
        
        # Apply configuration
        app.config.from_object(config_class)
        
        # Initialize configuration
        config_class.init_app(app)
        
        logger.info(f"[OK] Flask application created with {config_class.__name__}")
        return app
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to create Flask application: {e}")
        return None

def setup_signal_handlers(app):
    """Setup signal handlers for graceful shutdown"""
    logger = logging.getLogger(__name__)
    
    def signal_handler(signum, frame):
        logger.info(f"[STOP] Received signal {signum}, shutting down gracefully...")
        
        # Cleanup operations
        try:
            # Stop any running training
            from app import app_state
            if app_state.get('training_status', {}).get('is_training', False):
                app_state['training_status']['is_training'] = False
                logger.info("[STOP] Stopped running training processes")
            
            # Clear temporary files
            temp_dir = Path('temp')
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("[CLEAN] Cleaned up temporary files")
                
        except Exception as e:
            logger.error(f"[ERROR] Error during cleanup: {e}")
        
        logger.info("[DONE] Shutdown complete. Goodbye!")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)

def open_browser(url, delay=2):
    """Open web browser after delay"""
    def open_browser_delayed():
        try:
            webbrowser.open(url)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not open browser: {e}")
    
    Timer(delay, open_browser_delayed).start()

def print_banner():
    """Print application banner"""
    banner = f"""
    +------------------------------------------------------------------------------+
    |                                                                              |
    |                 QUADRATIC NEURAL NETWORK WEB APP                            |
    |                                                                              |
    |                      Advanced Neural Network Analysis                        |
    |                       for Quadratic Equations                                |
    |                                                                              |
    |                          Version: {VERSION_INFO['version']}                              |
    |                          Author: {VERSION_INFO['author']}                               |
    |                          Location: {VERSION_INFO['location']}                      |
    |                                                                              |
    |                      Beautiful Apple-like Design                             |
    |                                                                              |
    +------------------------------------------------------------------------------+
    """
    print(banner)

def print_startup_info(host, port, debug_mode):
    """Print startup information"""
    logger = logging.getLogger(__name__)
    
    info = f"""
    APPLICATION STARTUP INFORMATION
    {'=' * 50}
    
    Server Information:
       - Host: {host}
       - Port: {port}
       - Debug Mode: {'ON' if debug_mode else 'OFF'}
       - Environment: {os.environ.get('FLASK_ENV', 'development')}
    
    Access URLs:
       - Local: http://localhost:{port}
       - Network: http://{host}:{port}
    
    Application Features:
       - Dataset Management & Analysis
       - Multi-Scenario Neural Network Training
       - Interactive Prediction Interface
       - Advanced Performance Analysis
       - Model Comparison & Benchmarking
    
    Technical Stack:
       - Backend: Flask + Custom Neural Engine
       - Frontend: Modern JavaScript + Chart.js
       - Design: Apple-like UI/UX
       - AI Engine: Custom TensorFlow Alternative
    
    Quick Start:
       1. Open browser to http://localhost:{port}
       2. Upload quadratic equation dataset (CSV format)
       3. Select training scenarios
       4. Train neural network models
       5. Make predictions and analyze results
    
    Controls:
       - Press Ctrl+C to stop the server
       - Check logs in 'logs/quadratic_app.log'
       - Configuration in 'config.py'
    
    {'=' * 50}
    """
    
    print(info)
    logger.info("Application startup information displayed")

def validate_environment():
    """Validate environment and prerequisites"""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"[ERROR] Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    
    logger.info(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check Neural Engine
    if not check_neural_engine():
        return False
    
    # Check required directories
    required_dirs = ['templates', 'static', 'uploads', 'logs']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[DIR] Created directory: {dir_name}")
    
    logger.info("[OK] Environment validation complete")
    return True

def run_development_server(app, host='127.0.0.1', port=5000, debug=True, open_browser_flag=True):
    """Run development server"""
    logger = logging.getLogger(__name__)
    
    try:
        # Setup signal handlers
        setup_signal_handlers(app)
        
        # Print startup info
        print_startup_info(host, port, debug)
        
        # Open browser if requested
        if open_browser_flag:
            open_browser(f'http://localhost:{port}')
        
        logger.info(f"[START] Starting development server on {host}:{port}")
        logger.info("[READY] Application ready for connections")
        
        # Run the application
        app.run(host=host, port=port, debug=debug, use_reloader=False)
        
    except Exception as e:
        logger.error(f"[ERROR] Server startup failed: {e}")
        return False
    
    return True

def run_production_server(app, host='0.0.0.0', port=5000, workers=4):
    """Run production server using Gunicorn"""
    logger = logging.getLogger(__name__)
    
    try:
        import gunicorn.app.base
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    if key in self.cfg.settings and value is not None:
                        self.cfg.set(key.lower(), value)
            
            def load(self):
                return self.application
        
        options = {
            'bind': f'{host}:{port}',
            'workers': workers,
            'worker_class': 'sync',
            'timeout': 120,
            'keepalive': 2,
            'max_requests': 1000,
            'max_requests_jitter': 50,
            'access_logfile': 'logs/access.log',
            'error_logfile': 'logs/error.log',
            'log_level': 'info'
        }
        
        logger.info(f"[START] Starting production server with {workers} workers")
        print_startup_info(host, port, False)
        
        StandaloneApplication(app, options).run()
        
    except ImportError:
        logger.error("[ERROR] Gunicorn not available, falling back to development server")
        return run_development_server(app, host, port, debug=False, open_browser_flag=False)
    except Exception as e:
        logger.error(f"[ERROR] Production server startup failed: {e}")
        return False
    
    return True

def main():
    """Main application entry point"""
    # Print banner
    print_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quadratic Neural Network Web Application')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    parser.add_argument('--config', default='development', help='Configuration profile')
    
    args = parser.parse_args()
    
    # Set environment
    os.environ['FLASK_ENV'] = args.config
    
    # Get configuration
    config_class = get_config()
    
    # Setup logging
    logger = setup_logging(config_class)
    
    # Validate environment
    if not validate_environment():
        logger.error("[ERROR] Environment validation failed")
        sys.exit(1)
    
    # Create Flask application
    app = create_app()
    if not app:
        logger.error("[ERROR] Failed to create application")
        sys.exit(1)
    
    # Run server
    try:
        if args.production:
            logger.info("[PROD] Running in production mode")
            success = run_production_server(app, args.host, args.port, args.workers)
        else:
            logger.info("[DEV] Running in development mode")
            success = run_development_server(
                app, 
                args.host, 
                args.port, 
                args.debug, 
                not args.no_browser
            )
        
        if not success:
            logger.error("[ERROR] Server failed to start")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("[STOP] Server stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
