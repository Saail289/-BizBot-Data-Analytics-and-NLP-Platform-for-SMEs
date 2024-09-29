
from flask import Flask, request, redirect, url_for, send_from_directory, send_file,render_template_string, jsonify
import pandas as pd
import sweetviz as sv
from fpdf import FPDF
import os
from groq import Groq

app = Flask(__name__)

# Global variables
df = None
dashboard_path = None
llm_model = "llama3-8b-8192"
system_prompt = "Default system prompt"
chat_history = []  # To store chat history with LLM responses
conversation_memory_length = 5  # Default memory length

# Set your Groq API key
groq_api_key = "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"  # Replace with your actual API key
groq_client = Groq(api_key=groq_api_key)

# Ensure directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/css/custom.css')
def custom_static(filename):
    return send_from_directory('css', filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, dashboard_path, llm_model, system_prompt, chat_history, conversation_memory_length

    if request.method == 'POST':
        # Handle dataset upload
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                file_path = os.path.join('uploads', uploaded_file.filename)
                uploaded_file.save(file_path)
                try:
                    df = pd.read_csv(file_path)
                    generate_dashboard()
                except Exception as e:
                    print(f"Error reading file: {e}")
                    df = None
                    dashboard_path = None

        # Handle system prompt and LLM model selection
        system_prompt = request.form.get('system_prompt', system_prompt)
        llm_model = request.form.get('llm_model', llm_model)
        conversation_memory_length = int(request.form.get('memory_length', conversation_memory_length))

        # Handle user query
        if 'user_query' in request.form:
            user_query = request.form['user_query']
            llm_response = get_llm_response(user_query)
            chat_history.append({'user': user_query, 'llm': llm_response})  # Accuracy is optional

        return redirect(url_for('index'))

    return send_from_directory('','index.html')

def get_llm_response(query):
    # Simulated LLM response (replace with actual LLM logic)
    return "Simulated response for query: " + query

def generate_dashboard():
    global df, dashboard_path
    if df is not None:
        report = sv.analyze(df)
        report_path = os.path.join('static', 'sweetviz_report.html')
        report.show_html(report_path)
        dashboard_path = report_path

@app.route('/export_pdf')
def export_to_pdf():
    chat_history = session.get('chat_history', [])
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Chatbot Conversation Report", ln=True, align='C')

    for chat in chat_history:
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"You: {chat['human']}")
        pdf.ln(2)
        pdf.multi_cell(0, 10, txt=f"Chatbot: {chat['AI']}")
        pdf.ln(2)
        pdf.multi_cell(0, 10, txt=f"Accuracy: {chat['accuracy']}")

    pdf_file_path = 'static/chatbot_report.pdf'
    pdf.output(pdf_file_path)

    return send_file(pdf_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
