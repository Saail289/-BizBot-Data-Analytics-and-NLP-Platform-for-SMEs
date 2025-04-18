import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, send_from_directory, jsonify, send_file, session, redirect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import random  # Simulate accuracy scores
from fpdf import FPDF
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Set up directories
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Groq setup
groq_api_key = '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'
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if 'dataset_path' in session:
        dataset_path = session['dataset_path']
        df = pd.read_csv(dataset_path)
    else:
        df = None

    if request.method == 'POST':
        system_prompt = request.form.get("system_prompt", "")
        model = request.form.get("model", "llama3-8b-8192")
        conversational_memory_length = int(request.form.get("memory_length", 5))

        memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

        file = request.files.get('dataset')
        if file:
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)
            session['dataset_path'] = filepath
            df = pd.read_csv(filepath)

        user_input = request.form.get("user_input", "")

        if user_input:
            session['chat_history'].append({'human': user_input, 'AI': '', 'accuracy': ''})

            if "predict" in user_input.lower() or "model" in user_input.lower():
                return redirect("http://localhost:8501")

            if df is not None:
                dataset_info_message = f"Dataset information:\n{df.describe().to_string()}"
            else:
                dataset_info_message = "No dataset uploaded."

            response = ""
            if "plot" in user_input.lower():
                response = generate_plot(user_input, df)
            elif "dashboard" in user_input.lower():
                response = generate_dashboard(df)
            else:
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ])
                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=prompt,
                    verbose=True,
                    memory=memory
                )
                response = conversation.predict(human_input=f"{user_input}\n\n{dataset_info_message}")

            accuracy = random.uniform(0.7, 0.99)
            session['chat_history'][-1]['AI'] = response
            session['chat_history'][-1]['accuracy'] = f"{accuracy:.2%}"

        return render_template('index.html', chat_history=session['chat_history'])

    return send_from_directory(directory='.', path='index.html')

@app.route('/redirect_to_streamlit')
def redirect_to_streamlit():
    return redirect("http://localhost:8501")

# Route for exporting to PDF
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

# Function to generate plots
def generate_plot(user_input, df):
    if df is None:
        return "No dataset uploaded for visualization."

    if "heatmap" in user_input.lower():
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.savefig('static/heatmap.png')
        plt.close()
        return "Heatmap generated."

    elif "pairplot" in user_input.lower():
        sns.pairplot(df)
        plt.savefig('static/pairplot.png')
        plt.close()
        return "Pairplot generated."

    elif "histogram" in user_input.lower():
        column = request.form.get("column_name", "")
        if column in df.columns:
            df[column].hist(bins=30)
            plt.savefig('static/histogram.png')
            plt.close()
            return f"Histogram of {column} generated."
        else:
            return "Column not found."

    else:
        return "Plot type not recognized."

# Function to generate Sweetviz dashboard
def generate_dashboard(df):
    if df is None:
        return "No dataset uploaded for dashboard."
    
    report_path = 'static/sweetviz_report.html'
    
    report = sv.analyze(df)
    report.show_html(filepath=report_path, open_browser=False)
    
    if os.path.exists(report_path):
        print(f"Sweetviz report saved at: {report_path}")
        return "Sweetviz dashboard generated."
    else:
        print(f"Failed to generate Sweetviz report at: {report_path}")
        return "Error: Failed to generate Sweetviz dashboard."

if __name__ == '__main__':
    app.run(debug=True)