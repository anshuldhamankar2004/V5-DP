# Setup and Run Instructions

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space

### Required Software
- Git (for cloning repository)
- Python package manager (pip)
- Text editor or IDE

## Installation Steps

### 1. Clone Repository
```bash
git clone [REPOSITORY_URL]
cd "V5 DP - Copy (2)"
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your configuration
# Add your API keys and configuration values
```

### 5. Create Required Directories
```bash
mkdir tmp
mkdir tmp\uploads
mkdir tmp\saved_models
mkdir tmp\notebooks
mkdir tmp\sessions
```

## Environment Variables Setup

### Required Variables
Edit your `.env` file with the following:

```env
# Flask Configuration
SECRET_KEY=your_secret_key_here
DEBUG=True

# API Keys
TOGETHER_API_KEY=your_together_ai_api_key

# Database Configuration (if using external DB)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# File Upload Settings
MAX_FILE_SIZE=104857600  # 100MB in bytes
ALLOWED_EXTENSIONS=csv,xlsx

# Session Configuration
SESSION_TIMEOUT=3600  # 1 hour in seconds
```

## Database Setup (Optional)

### For External Database Connection
1. **PostgreSQL Setup**:
```bash
# Install PostgreSQL
# Create database and user
createdb your_database
createuser your_username
```

2. **MySQL Setup**:
```bash
# Install MySQL
# Create database and user
mysql -u root -p
CREATE DATABASE your_database;
CREATE USER 'your_username'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON your_database.* TO 'your_username'@'localhost';
```

## Running the Application

### Development Mode
```bash
# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Run Flask application
python app.py
```

### Production Mode
```bash
# Set production environment
export FLASK_ENV=production

# Run with Gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Accessing the Application

### Local Development
- **URL**: http://localhost:5000
- **Upload Page**: http://localhost:5000/
- **API Base**: http://localhost:5000/api

### Testing the Setup
1. Navigate to http://localhost:5000
2. Upload a sample CSV file
3. Verify file processing works
4. Test basic data operations

## Troubleshooting

### Common Issues

**Port Already in Use**:
```bash
# Kill process on port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

**Missing Dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Permission Errors**:
```bash
# Ensure proper directory permissions
chmod 755 tmp/
chmod 755 tmp/uploads/
```

### Log Files
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`
- Check console output for real-time debugging

## Development Setup

### Additional Development Tools
```bash
# Install development dependencies
pip install pytest flask-testing black flake8

# Run tests
pytest

# Code formatting
black .

# Linting
flake8 .
```