# project/app.py
from flask import Flask ,render_template
from flask_cors import CORS
from config import SECRET_KEY, UPLOAD_FOLDER

# Initialize app
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

# === Register Blueprints from routes/ ===
from routes.upload import upload_bp
from routes.data_details import details_bp
from routes.describe import describe_bp
from routes.cleaning import cleaning_bp
from routes.modeling import modeling_bp
from routes.visualize import visualize_bp
from routes.data_access import data_access_bp
from routes.db_connect import db_connect_bp
from services.save_nb import notebook_bp
from routes.custom_clean import custom_clean_bp
from routes.custom_clean import api_clean_bp

app.register_blueprint(custom_clean_bp)
app.register_blueprint(notebook_bp)
app.register_blueprint(db_connect_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(describe_bp)
app.register_blueprint(cleaning_bp)
app.register_blueprint(modeling_bp, url_prefix="/modeling")
app.register_blueprint(visualize_bp)
app.register_blueprint(data_access_bp)
app.register_blueprint(api_clean_bp)
app.register_blueprint(details_bp)

@app.route('/')
def upload_page():
    """Renders the upload page."""
    return render_template('uploads.html')

# Run server

if __name__ == '__main__':
    app.run(debug=True)