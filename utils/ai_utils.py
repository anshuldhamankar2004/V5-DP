# utils/ai_utils.py

import os
import re
import traceback
from datetime import datetime
from spellchecker import SpellChecker
from config import NOTEBOOK_DIR
from notebook_logger import append_to_notebook
import together

os.environ['TOGETHER_API_KEY'] = "65e09cf3aee49bbe53caab00364984b80cf19e7f83af69d69bf51ce36d46d485"
client = together.Together()

MAX_RETRIES = 3

def spell_check_sql(query):
    if not query:
        return ""
    spell = SpellChecker()
    words = query.split()
    misspelled = spell.unknown(words)
    return " ".join([spell.correction(w) if w in misspelled else w for w in words])


from together import Together

client = Together()  # Automatically picks TOGETHER_API_KEY

def ai(prompt, columns, query_language, file_id=None):
    formatted_columns = ", ".join([f"`{col}`" for col in columns])

    language_context = ""
    if query_language == "sql":
        language_context = (
            "You are a SQL expert. Generate a valid SQL query to modify a table named 'data' with columns: [{formatted_columns}] based on the user's request.\n"
            "The SQL query should ideally use an `UPDATE` statement to modify the `data` table.\n"
            "NEVER use invalid or non-existent columns. Use only the provided columns.\n"
            "Very strict warning: Don't explain anything. DO NOT output markdown. Print the SQL query only once. DO NOT use ``` or extra formatting. DO NOT add any comments or extra information.\n"
            "Return 'modification' on the first line, followed by the SQL code on the next line."
        )
    else: # Default and enforce Python/Pandas
        language_context = (
            "You are a Python expert and Advanced Data Scientist. Your goal is to generate Python/Pandas code to modify a DataFrame called `df` based on the user's request.\n"
            "NEVER use invalid or non-existent columns. Use only the provided columns: [{formatted_columns}].\n"
            "We use pandas as `pd`, numpy as `np`. Ensure your code is valid and executable.\n"
            "Very strict warning: Don't explain anything. DO NOT output markdown. Print the code only once. DO NOT use ``` or extra formatting. DO NOT import anything. DO NOT use emojis or comments.\n"
            "Return 'modification' on the first line, followed by the Pandas code on the next line."
        )

    formatted_prompt = (
        f"{language_context}\n"
        f"User query: {prompt}\n"
        f"Return the query type ('analysis', 'modification', 'visualization') on the first line.\n"
        f"On the second line, return only the clean executable code (or SQL query).\n"
        f"Do not explain, comment, or return anything else.\n"
    )

    try:
        print("\n🧪 ========= AI DEBUG START =========")
        print("🕒 Timestamp:", datetime.now())
        print("🧠 Model:", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
        print("📤 Prompt Sent:\n", formatted_prompt)
        print("📁 File ID:", file_id or "N/A")
        print("🌍 Query Language:", query_language)
        print("====================================")

        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
            top_p=0.5,
            max_tokens=512
        )

        ai_response = response.choices[0].message.content.strip()

        print("🧠 Raw AI Response:\n", ai_response)

        if not ai_response:
            return "", ""

        lines = [line.strip() for line in ai_response.splitlines() if line.strip()]
        if len(lines) < 2:
            return "", ""

        query_type = lines[0].lower()
        code_lines = [line for line in lines[1:] if not line.lower().startswith(query_type)]
        clean_code = "\n".join(code_lines).split("```")[0].strip()

        if query_language == "python" and file_id:
            notebook_path = f"{NOTEBOOK_DIR}/{file_id}_session.ipynb"
            append_to_notebook(notebook_path, f"# Natural Language Query:\n# {prompt}")
            append_to_notebook(notebook_path, clean_code)

        return query_type, clean_code

    except Exception as e:
        print(f"❌ AI Error: {str(e)}")
        traceback.print_exc()
        return "", ""