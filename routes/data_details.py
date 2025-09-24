# data_details.py
from flask import Blueprint, jsonify, render_template, request
import pandas as pd
from services.session_manager import load_session
import numpy as np
from io import StringIO
from collections import Counter
import time

import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import datetime
import together
import traceback

os.environ['TOGETHER_API_KEY'] = "65e09cf3aee49bbe53caab00364984b80cf19e7f83af69d69bf51ce36d46d485"
client = together.Together()
details_bp = Blueprint('data_details', __name__)


@details_bp.route('/data_details/<file_id>', methods=['GET'])
def data_details(file_id):
    """Renders the data details page."""
    session_data = load_session(file_id)
    if not session_data:
        return "Error: File ID not found", 404
    return render_template("data_details.html", file_id=file_id)



@details_bp.route('/data_details/<file_id>/_get_dataframe', methods=['GET'])
def _get_dataframe(file_id):
    """Helper function to load DataFrame from session with error handling."""
    session_data = load_session(file_id)
    if not session_data or 'modified_parquet' not in session_data:
        return None, "Error: DataFrame not found in session."
    try:
        parquet_buffer = io.BytesIO(session_data['modified_parquet'])
        df = pd.read_parquet(parquet_buffer)
        return df, None
    except Exception as e:
        return None, f"Error loading DataFrame from session: {str(e)}"





@details_bp.route('/data_details/<file_id>/overview', methods=['GET'])
def overview(file_id):
    """Returns an overview of the dataset."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        buffer = StringIO()
        df.info(buf=buffer, memory_usage='deep')
        s = buffer.getvalue().splitlines()
        memory_line = next((line for line in s if "memory usage" in line), "N/A")
        memory_usage = memory_line.split(':')[-1].strip() if "N/A" not in memory_line else "N/A"
        return jsonify({
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_usage": memory_usage
        })
    except Exception as e:
        return jsonify({"error": f"Error generating overview: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/preview', methods=['GET'])
def preview(file_id):
    """Returns a preview of the dataset."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    limit = int(request.args.get('limit', 5))
    try:
        preview_data = df.head(limit).fillna('').to_dict(orient='records')
        columns = df.columns.tolist()
        return jsonify({"columns": columns, "data": preview_data})
    except Exception as e:
        return jsonify({"error": f"Error generating preview: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/data_types', methods=['GET'])
def data_types(file_id):
    """Returns a summary of column data types."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        data_types_summary = {}
        for col in df.columns:
            unique_values = df[col].nunique()
            missing_count = df[col].isnull().sum()
            zero_count = (df[col] == 0).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
            data_types_summary[col] = {
                "dtype": str(df[col].dtype),
                "unique_values": int(unique_values),
                "missing_percentage": f"{(missing_count / len(df) * 100):.2f}%" if len(df) > 0 else "0.00%",
                "zero_count": int(zero_count)
            }
        return jsonify(data_types_summary)
    except Exception as e:
        return jsonify({"error": f"Error getting data types: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/descriptive_stats', methods=['GET'])
def descriptive_stats(file_id):
    """Returns descriptive statistics for numerical columns."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        numeric_df = df.select_dtypes(include=['number'])
        stats = numeric_df.describe().fillna('').to_dict()
        # Format the stats for better presentation in the frontend
        formatted_stats = {}
        for col, values in stats.items():
            formatted_stats[col] = {k: f"{v:.2f}" if isinstance(v, (int, float)) else v for k, v in values.items()}
        return jsonify(formatted_stats)
    except Exception as e:
        return jsonify({"error": f"Error getting descriptive statistics: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/outliers', methods=['GET'])
def outliers(file_id):
    """Returns a summary of potential outliers using the IQR method."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        numeric_df = df.select_dtypes(include=['number'])
        outlier_summary = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
            outlier_summary[col] = int(outliers_count)
        return jsonify(outlier_summary)
    except Exception as e:
        return jsonify({"error": f"Error getting outlier summary: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/categorical_summary', methods=['GET'])
def categorical_summary(file_id):
    """Returns a summary of categorical columns."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        categorical_df = df.select_dtypes(include=['object', 'category'])
        categorical_summary = {}
        for col in categorical_df.columns:
            value_counts = df[col].value_counts()
            top_value = value_counts.index[0] if not value_counts.empty else None
            frequency = int(value_counts.iloc[0]) if not value_counts.empty else 0
            unique_values = df[col].nunique()
            categorical_summary[col] = {
                "unique_values": int(unique_values),
                "top_value": str(top_value) if top_value is not None else None,
                "frequency": frequency
            }
        return jsonify(categorical_summary)
    except Exception as e:
        return jsonify({"error": f"Error getting categorical summary: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/correlation_matrix', methods=['GET'])
def correlation_matrix(file_id):
    """Returns the correlation matrix for numerical columns."""
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        numeric_df = df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr().fillna('').to_dict()
        return jsonify(correlation_matrix)
    except Exception as e:
        return jsonify({"error": f"Error getting correlation matrix: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/data_quality', methods=['GET'])
def data_quality(file_id):
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        quality_summary = []
        for col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            is_constant = df[col].nunique(dropna=False) == 1
            dtype = str(df[col].dtype)
            duplicate_count = df[col].duplicated().sum()
            quality_summary.append({
                "column": col,
                "missing_percent": f"{missing_pct:.2f}%",
                "duplicate": "Yes" if duplicate_count > 0 else "No",
                "constant": "Yes" if is_constant else "No",
                "dtype": dtype
            })
        return jsonify(quality_summary)
    except Exception as e:
        return jsonify({"error": f"Error computing data quality: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/distribution_charts', methods=['GET'])
def distribution_charts(file_id):
    df, error = _get_dataframe(file_id)
    if error:
        return jsonify({"error": error}), 404
    try:
        charts_html = ""
        for col in df.select_dtypes(include=['number']).columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[col].hist(ax=ax, bins=10, edgecolor='black')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            buffer = io.BytesIO()
            canvas = FigureCanvas(fig)
            canvas.draw()
            image_stream = io.BytesIO()
            fig.savefig(image_stream, format='png')
            image_stream.seek(0)
            img_data = base64.b64encode(image_stream.read()).decode('utf-8')
            charts_html += f'<img src="data:image/png;base64,{img_data}" alt="Distribution of {col}"><br>'
            plt.close(fig)

        for col in df.select_dtypes(include=['object', 'category']).columns:
            top_n = 10
            value_counts = df[col].value_counts().nlargest(top_n)
            if not value_counts.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                value_counts.plot(kind='bar', ax=ax, edgecolor='black')
                ax.set_title(f'Top {top_n} Values in {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                buffer = io.BytesIO()
                canvas = FigureCanvas(fig)
                canvas.draw()
                image_stream = io.BytesIO()
                fig.savefig(image_stream, format='png')
                image_stream.seek(0)
                img_data = base64.b64encode(image_stream.read()).decode('utf-8')
                charts_html += f'<img src="data:image/png;base64,{img_data}" alt="Top values of {col}"><br>'
                plt.close(fig)

        return jsonify({"html": charts_html})
    except Exception as e:
        return jsonify({"error": f"Error generating distribution charts: {str(e)}"}), 500


@details_bp.route('/data_details/<file_id>/datetime_analysis', methods=['GET'])
def datetime_analysis(file_id):
    df, error = _get_dataframe(file_id)
    if error:
        print(f"Datetime Analysis Error: {error}")
        return jsonify({"error": error}), 404
    try:
        print("Starting Datetime Analysis")
        print(f"DataFrame Dtypes:\n{df.dtypes}")
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        print(f"Datetime Columns Found: {list(datetime_cols)}")
        date_info = {}
        for col in datetime_cols:
            series = df[col].dropna()
            if series.empty:
                print(f"Datetime Column '{col}' is empty after dropping NaNs.")
                continue
            min_date = series.min().isoformat()
            max_date = series.max().isoformat()
            # Attempt to infer granularity
            diffs = series.diff().dropna()
            if not diffs.empty:
                most_common_diff = diffs.mode()
                granularity = str(most_common_diff[0]) if not most_common_diff.empty else "Unknown"
                print(f"Datetime Column '{col}': Min='{min_date}', Max='{max_date}', Granularity='{granularity}'")
            else:
                granularity = "Single Value"
                print(
                    f"Datetime Column '{col}': Min='{min_date}', Max='{max_date}', Granularity='{granularity}' (Single Value or no difference)")

            date_info[col] = {
                "min_date": min_date,
                "max_date": max_date,
                "granularity": granularity
            }
        print(f"Datetime Analysis Result: {date_info}")
        return jsonify(date_info)
    except Exception as e:
        error_message = f"Error in datetime analysis: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500


@details_bp.route('/data_details/<file_id>/text_summary', methods=['GET'])
def text_summary(file_id):
    df, error = _get_dataframe(file_id)
    if error:
        print(f"Text Summary Error: {error}")
        return jsonify({"error": error}), 404
    try:
        print("Starting Text Summary")
        print(f"DataFrame Dtypes:\n{df.dtypes}")
        object_cols = df.select_dtypes(include=['object']).columns
        print(f"Object Columns Found: {list(object_cols)}")
        summary = {}
        for col in object_cols:
            text_series = df[col].dropna().astype(str)
            if text_series.empty:
                print(f"Text Column '{col}' is empty after dropping NaNs.")
                continue
            text_lengths = text_series.apply(len)
            all_words = ' '.join(text_series.str.lower()).split()
            word_counts = Counter(all_words)
            top_keyword = word_counts.most_common(1)[0][0] if word_counts else "N/A"
            avg_length = float(text_lengths.mean()) if not text_lengths.empty else 0.0
            max_length = int(text_lengths.max()) if not text_lengths.empty else 0
            print(
                f"Text Column '{col}': Avg Length='{avg_length}', Max Length='{max_length}', Top Keyword='{top_keyword}'")
            summary[col] = {
                "avg_length": avg_length,
                "max_length": max_length,
                "top_keyword": top_keyword
            }
        print(f"Text Summary Result: {summary}")
        return jsonify(summary)
    except Exception as e:
        error_message = f"Error generating text summary: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500


@details_bp.route('/data_details/<file_id>/ai_suggestions', methods=['POST'])
def ai_cleanup_suggestions(file_id):
    df, error = _get_dataframe(file_id)
    if error:
        print(f"AI Suggestions Error: {error}")
        return jsonify({"error": error}), 404
    try:
        all_analysis_data = request.get_json()
        print("Received analysis data for AI suggestions:", all_analysis_data)

        language_context = (
            "You are a highly skilled Python expert and Advanced Data Scientist. Your primary goal is to provide precise and actionable data *cleaning* and *preprocessing* suggestions, along with detailed explanations.  Focus ONLY on steps that improve data quality. Do not provide any code.\n"
            "If, for any reason, you cannot provide meaningful data cleaning suggestions, respond with ONLY the phrase 'No suggestions available.' and nothing else.\n"
            "Your output should be structured as follows:\n"
            "    -   The first line should contain the string 'Suggestions:'.\n"
            "    -   The subsequent lines should contain a numbered list with at least 10 detailed explanations of data cleaning and preprocessing steps, including why each step is necessary for data quality. Focus on explaining how the suggestions address issues like missing values, outliers, incorrect data types, and inconsistencies. Provide context for each suggestion. Do not include any code.\n"
            "If no suggestions are available, the first line should contain 'No suggestions available.' and nothing else."
        )

        prompt = f"""
                            {language_context}

                            Based on the following comprehensive analysis of the dataset, provide specific and actionable data *cleaning* and *preprocessing* suggestions. Focus ONLY on modifications to improve data quality and prepare the data for further analysis or modeling. Do not suggest exploratory analysis, feature engineering, or model training. Do not provide any code.

                            Dataset Overview:
                            {all_analysis_data.get('overview', {})}

                            Column Data Types and Characteristics:
                            {all_analysis_data.get('data_types', {})}

                            Descriptive Statistics:
                            {all_analysis_data.get('descriptive_stats', {})}

                            Outlier Summary:
                            {all_analysis_data.get('outliers', {})}

                            Categorical Column Summary:
                            {all_analysis_data.get('categorical_summary', {})}

                            Data Quality Assessment:
                            {all_analysis_data.get('data_quality', {})}

                            Datetime Analysis:
                            {all_analysis_data.get('datetime_analysis', {})}

                            Text Column Summary:
                            {all_analysis_data.get('text_summary', {})}
                            """
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    top_p=0.5,
                    max_tokens=1024,
                )
                ai_message = response.choices[0].message.content.strip()

                if "Suggestions:" in ai_message:
                    suggestion_lines = ai_message.split('\n')
                    if len(suggestion_lines) >= 11 and "1." in suggestion_lines[1]:
                        break
                    else:
                        print(f"Attempt {attempt + 1} failed with invalid response: {ai_message}")
                        time.sleep(retry_delay)
                elif "No suggestions available." in ai_message:
                    break
                else:
                    # Handle the case where "Suggestions:" is missing but there are suggestions
                    suggestion_lines = ai_message.split('\n')
                    if len(suggestion_lines) >= 10 and "1." in suggestion_lines[0]:
                        ai_message = "Suggestions:\n" + ai_message
                        break
                    else:
                        print(f"Attempt {attempt + 1} failed with invalid response: {ai_message}")
                        time.sleep(retry_delay)

            except Exception as e:
                error_message = f"Error generating AI suggestions: {str(e)}\n{traceback.format_exc()}"
                print(error_message)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return jsonify({"error": error_message}), 500

        # Return
        if "Suggestions:" in ai_message:
            return jsonify({"suggestion": ai_message})
        elif "No suggestions available." in ai_message:
            return jsonify({"suggestion": "No suggestions available."})
        else:
            return jsonify({"suggestion": "No suggestions available."})

    except Exception as e:
        error_message = f"Error generating AI suggestions: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({"error": error_message}), 500
