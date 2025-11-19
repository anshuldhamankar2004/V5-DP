# System Architecture

## System Architecture Overview

The application follows a modular Flask-based architecture with clear separation of concerns. The system is designed as a web-based data processing and analytics platform with AI-powered capabilities.


## Component Breakdown

### Core Application Layer
- **app.py**: Main Flask application entry point
- **config.py**: Centralized configuration management
- **core.py**: Core functionality and utilities

### Route Layer (`/routes`)
- **upload.py**: File upload handling
- **data_details.py**: Data inspection and metadata
- **describe.py**: Statistical analysis endpoints
- **cleaning.py**: Data cleaning operations
- **modeling.py**: Machine learning model operations
- **visualize.py**: Data visualization endpoints
- **data_access.py**: Data retrieval operations
- **db_connect.py**: Database connectivity
- **custom_clean.py**: Custom cleaning operations

### Service Layer (`/services`)
- **session_manager.py**: User session handling
- **data_loader.py**: Data loading utilities
- **db_loader.py**: Database integration
- **model_trainer.py**: ML model training
- **model_loader.py**: Model persistence
- **automl_recommender.py**: AutoML recommendations
- **schema_generator.py**: Database schema generation
- **save_nb.py**: Notebook generation
- **explainers.py**: Model explanation services

### Utility Layer (`/utils`)
- **ai_utils.py**: AI/LLM integration utilities
- **logging.py**: Application logging
- **sanitizer.py**: Input validation and sanitization

### Presentation Layer (`/templates`, `/static`)
- **HTML templates**: User interface components
- **CSS styling**: Application styling
- **Static assets**: Images and resources

## Data Flow Explanation

1. **Data Ingestion**: Users upload files or connect to databases
2. **Session Management**: Each interaction creates a unique session
3. **Data Processing**: Raw data is processed through cleaning pipelines
4. **AI Integration**: Natural language queries are processed via LLM
5. **Analysis & Modeling**: Statistical analysis and ML model training
6. **Visualization**: Results are presented through web interface
7. **Export**: Processed data and notebooks can be exported