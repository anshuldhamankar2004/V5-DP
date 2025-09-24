# utils/sanitizer.py
from core import *
from config import UPLOAD_FOLDER

def sanitize_filename(filename):
    sanitized = secure_filename(filename)
    max_length = 100
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext

    unique_id = str(uuid.uuid4())[:8]
    sanitized = f"{unique_id}_{sanitized}"
    sanitized = os.path.normpath(sanitized)

    print(f"✅ Sanitized filename: {sanitized}")
    return sanitized

def convert_nan_to_none(obj):
    """Recursively converts NaN, Infinity, and -Infinity to None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(x) for x in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


def escape_valid_columns(query, valid_columns):
    """
    Properly escapes only valid column names in the SQL query.
    - Ensures no double escaping issues.
    - Avoids modifying already escaped columns.
    """
    # ✅ Precompile regex for performance
    column_pattern = re.compile(r'`?(\w+)`?')

    def escape_column(match):
        col = match.group(1)
        if col in valid_columns:
            # ✅ Only add backticks if not already escaped
            return f"`{col}`" if not query.startswith("ALTER") else col
        return col

    # ✅ Apply proper escaping without double backticks
    escaped_query = column_pattern.sub(escape_column, query)

    print(f"🔥 Properly Escaped Query: {escaped_query}")
    return escaped_query

def escape_sql_columns(query, columns):
    """
    Automatically escapes SQL column names with backticks to avoid syntax errors.
    """
    if not query or not columns:
        return query

    # ✅ Escape only valid column names
    for col in columns:
        # Escape column names with backticks if they contain underscores or special characters
        escaped_col = f"`{col}`" if "_" in col or col.lower() in ["date", "user", "group"] else col
        query = re.sub(rf'\b{re.escape(col)}\b', escaped_col, query, flags=re.IGNORECASE)

    return query

def allowed_file(filename, allowed_extensions={'csv', 'xlsx'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
