# APIs and Models Documentation

## Backend API Endpoints

### File Upload & Management
```
POST /upload
- Description: Upload CSV/Excel files
- Request: multipart/form-data with file
- Response: {"success": true, "file_id": "uuid"}
```

### Database Connectivity
```
POST /connect
- Description: Connect to external databases
- Request: {"dbType": "mysql", "host": "localhost", "port": 3306, "username": "user", "password": "pass", "database": "db", "table": "table"}
- Response: {"success": true, "file_id": "uuid"}
```

### Data Operations
```
GET /data/details/{file_id}
- Description: Get data metadata and statistics
- Response: {"columns": [], "shape": [rows, cols], "dtypes": {}}

POST /data/describe/{file_id}
- Description: Statistical analysis of dataset
- Response: {"statistics": {}, "summary": ""}

POST /data/clean/{file_id}
- Description: Apply data cleaning operations
- Request: {"operations": [], "parameters": {}}
- Response: {"success": true, "modified_data": []}
```

### AI-Powered Analysis
```
POST /ai/query/{file_id}
- Description: Natural language data queries
- Request: {"query": "string", "language": "python|sql"}
- Response: {"query_type": "analysis|modification|visualization", "code": "string"}
```

### Modeling Endpoints
```
POST /modeling/train/{file_id}
- Description: Train ML models
- Request: {"target_column": "string", "model_type": "classification|regression"}
- Response: {"model_id": "uuid", "metrics": {}}

GET /modeling/results/{model_id}
- Description: Get model performance metrics
- Response: {"accuracy": 0.95, "metrics": {}}
```

## ML Model Overview

### Primary AI Model
- **Model**: Meta-Llama/Llama-4-Maverick-17B-128E-Instruct-FP8
- **Provider**: Together AI
- **Purpose**: Natural language to code generation
- **Input**: Natural language queries + data context
- **Output**: Python/SQL code for data operations

### AutoML Integration
- **Framework**: Custom AutoML recommender
- **Supported Tasks**: Classification, Regression
- **Model Selection**: Automated based on data characteristics
- **Evaluation**: Cross-validation with multiple metrics

## Database Schema Placeholder

```sql
-- Sessions Table
CREATE TABLE sessions (
    id VARCHAR(36) PRIMARY KEY,
    file_path TEXT,
    created_at TIMESTAMP,
    modified_at TIMESTAMP
);

-- Models Table
CREATE TABLE models (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36),
    model_type VARCHAR(50),
    metrics JSON,
    created_at TIMESTAMP
);
```

## Request/Response Formats

### Standard Success Response
```json
{
    "success": true,
    "data": {},
    "message": "Operation completed successfully"
}
```

### Standard Error Response
```json
{
    "success": false,
    "error": "Error description",
    "code": "ERROR_CODE"
}
```