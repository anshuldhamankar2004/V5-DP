# DataPilot.ai - AI-Powered Data Analytics Platform

## Overview

This project is an intelligent data analytics platform that democratizes data science by enabling users to perform complex data analysis through natural language queries. The platform combines the power of AI with intuitive web interfaces to make data science accessible to non-technical users while maintaining the flexibility needed by data professionals.

**Key Innovation**: Natural language to code generation for data operations, eliminating the need for programming expertise in data analysis workflows.

## Tech Stack

### Core Technologies
- **Backend**: Python 3.x, Flask 2.x
- **Data Processing**: Pandas, Polars, NumPy
- **AI/ML**: Together AI (Llama-4-Maverick), AutoML
- **Database**: Multi-database support (MySQL, PostgreSQL, SQLite)
- **Frontend**: HTML5, CSS3, JavaScript

### Key Libraries
- Flask-CORS, python-dotenv, SpellChecker
- UUID for session management
- Jupyter notebook integration

## Architecture

The platform follows a modular Flask-based architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │────│   Flask Routes   │────│   AI Services   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │  Data Processing │
                       │   (Pandas/Polars) │
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │  Session Manager │
                       │  & File Storage  │
                       └──────────────────┘
```

**See**: [Architecture.md](Docs/Architecture.md) for detailed system design

## Features

### 🚀 Core Capabilities
- **Multi-format Data Upload**: CSV, Excel support with validation
- **Database Connectivity**: Direct connection to MySQL, PostgreSQL, SQLite
- **Natural Language Querying**: Plain English to Python/SQL code generation
- **Intelligent Data Cleaning**: Automated detection and cleaning of data issues
- **AutoML Integration**: Automated model training and evaluation

### 🧠 AI-Powered Features
- **Code Generation**: Natural language to executable code
- **Statistical Analysis**: Automated descriptive statistics and profiling
- **Predictive Modeling**: Classification and regression with explainability
- **Smart Visualizations**: Auto-generated charts based on data characteristics

### 🔧 Productivity Tools
- **Session Management**: Persistent workflows across browser sessions
- **Jupyter Notebook Export**: Reproducible analysis documentation
- **Real-time Feedback**: Live previews and validation
- **Error Recovery**: Graceful handling of failures with retry mechanisms

**See**: [Features_Implemented.md](Docs/Features_Implemented.md) for complete feature list

## Setup Instructions

### Quick Start
```bash
# 1. Clone and navigate to project
git clone [REPOSITORY_URL]
cd "V5 DP - Copy (2)"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run application
python app.py
```

### Access Application
- **Local URL**: http://localhost:5000
- **Upload Interface**: http://localhost:5000/
- **API Documentation**: See [APIs_and_Models.md](Docs/APIs_and_Models.md)

**See**: [Setup_and_Run.md](Docs/Setup_and_Run.md) for detailed installation guide

## Future Scope

### Immediate Enhancements (Next 3 months)
- **Performance Optimization**: Async processing, caching layer
- **Advanced UI**: React-based frontend with improved UX
- **User Authentication**: Multi-user support with role-based access
- **Cloud Deployment**: Containerized AWS/Azure deployment

### Long-term Vision (6+ months)
- **Enterprise Features**: SSO, audit logs, compliance reporting
- **Advanced Analytics**: Time series forecasting, anomaly detection
- **Collaboration Tools**: Real-time multi-user editing
- **AI Assistant**: Conversational interface for data exploration

**See**: [Whats_Next.md](Docs/Whats_Next.md) for complete roadmap

## Technical Highlights

### Innovation Points
- **Natural Language Interface**: Breakthrough in making data science accessible
- **Hybrid Processing**: Combines Pandas compatibility with Polars performance
- **AI-Driven Automation**: Reduces manual data preparation by 80%
- **Modular Architecture**: Scalable design for enterprise deployment

### Performance Metrics
- **Response Time**: <10 seconds for complex AI queries
- **Data Capacity**: Optimized for datasets up to 1M rows
- **Accuracy**: 85% success rate for executable code generation
- **User Productivity**: 70% reduction in analysis time

**See**: [Impact_and_Metrics.md](Docs/Impact_and_Metrics.md) for detailed metrics

## Documentation Structure

```
/Submission/Docs/
├── TechStack.md              # Complete technology overview
├── Architecture.md           # System design and components
├── APIs_and_Models.md        # API documentation and ML models
├── Features_Implemented.md   # Comprehensive feature list
├── Setup_and_Run.md         # Installation and deployment guide
├── Problem_Statement.md     # Problem analysis and solution approach
├── Impact_and_Metrics.md    # Performance metrics and user impact
├── Tradeoffs_and_Decisions.md # Technical decisions and rationale
└── Whats_Next.md           # Future roadmap and improvements
```

## Credits

**Development Team**: Anshul Dhamankar
**AI Integration**: Together AI (Llama-4-Maverick-17B-128E-Instruct-FP8)
**Framework**: Flask ecosystem and Python data science community
**Inspiration**: Democratizing data science for non-technical users

---

**Repository**: https://github.com/anshuldhamankar2004/V5-DP
**License**: MIT	
