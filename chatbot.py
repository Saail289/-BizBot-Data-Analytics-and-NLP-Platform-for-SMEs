import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, send_from_directory, jsonify, send_file, session, redirect, url_for, flash
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Set up directories
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Groq setup
groq_api_key = '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'
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

# Route for the main chatbot page
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []

    # Load dataset from session if it exists
    if 'dataset_path' in session:
        dataset_path = session['dataset_path']
        df = pd.read_csv(dataset_path)
    else:
        df = None

    if request.method == 'POST':
        system_prompt = request.form.get("system_prompt", "")
        model = request.form.get("model", "llama3-8b-8192")
        conversational_memory_length = int(request.form.get("memory_length", 5))

        memory = ConversationBufferWindowMemory(
            k=conversational_memory_length, 
            memory_key="chat_history", 
            return_messages=True
        )

        # File upload handling
        file = request.files.get('dataset')
        if file:
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)
            session['dataset_path'] = filepath  # Save dataset path in session
            df = pd.read_csv(filepath)
            flash('Dataset uploaded successfully!', 'success')

        user_input = request.form.get("user_input", "")

        if user_input:
            # Add user question to chat history
            session['chat_history'].append({'human': user_input, 'AI': '', 'accuracy': ''})

            # Construct the LLM conversation chain
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ]
            )

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory
            )

            # Process dataset and user query
            if df is not None:
                dataset_info_message = f"Dataset information:\n{df.describe().to_string()}"
            else:
                dataset_info_message = "No dataset uploaded."

            # Check if user asked for visualizations or model predictions
            response = ""
            if "plot" in user_input.lower():
                response = generate_plot(user_input, df)
            elif "dashboard" in user_input.lower():
                response = generate_dashboard(df)
            elif "predict" in user_input.lower():
                # Redirect to prediction page
                return redirect(url_for('model_predictions_page'))
            else:
                # Normal chatbot response
                response = conversation.predict(human_input=f"{user_input}\n\n{dataset_info_message}")

            # Simulate accuracy
            accuracy = random.uniform(0.7, 0.99)
            session['chat_history'][-1]['AI'] = response
            session['chat_history'][-1]['accuracy'] = f"{accuracy:.2%}"

        return render_template('index_chat.html', chat_history=session['chat_history'])

    # Serve index.html directly from the main directory
    return render_template('index.html')

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

def train_model(X, y, model_choice, param_grid):
    # Define the preprocessor for categorical and numerical data
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', SimpleImputer(strategy='mean'), X.columns.difference(categorical_cols))
        ],
        remainder='passthrough'
    )

    # Model selection based on user's choice
    model = None
    scoring_metric = 'accuracy'

    if model_choice == "Linear Regression":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        scoring_metric = 'r2'
    elif model_choice == "Logistic Regression":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
    elif model_choice == "Random Forest Classifier":
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ])
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    # Add other models similarly...

    # Perform GridSearchCV or just fit if no hyperparameters
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring_metric)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
    else:
        model.fit(X, y)
        best_model = model
        best_score = cross_val_score(model, X, y, cv=5, scoring=scoring_metric).mean()
        best_params = {}

    return best_model, best_params, best_score

@app.route('/predict', methods=['GET', 'POST'])
def model_predictions_page():
    if 'uploaded_file_path' not in session:
        session['uploaded_file_path'] = None

    prediction_results = None
    best_params = None
    best_score = None
    accuracy = None
    prediction = None
    df = None  # Initialize df to None

    if request.method == 'POST':
        # Handling file upload
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                filepath = os.path.join('static/uploads', uploaded_file.filename)
                uploaded_file.save(filepath)
                session['uploaded_file_path'] = filepath
                flash('Dataset uploaded successfully!', 'success')
        elif 'train_model' in request.form:
            try:
                # Load dataset
                filepath = session.get('uploaded_file_path', None)
                if not filepath or not os.path.exists(filepath):
                    flash('No dataset uploaded!', 'danger')
                    return redirect(request.url)
                
                df = pd.read_csv(filepath)

                # Get target column and model choice
                target_column = request.form.get('target_column')
                model_choice = request.form.get('model_choice')

                if not target_column or not model_choice:
                    flash('Please select target column and model.', 'warning')
                    return redirect(request.url)

                # Preparing the data
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Identifying categorical columns
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

                # Preprocessing pipeline
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                        ('num', SimpleImputer(strategy='mean'), X.columns.difference(categorical_cols))
                    ],
                    remainder='passthrough'
                )

                # Model selection and parameter grid
                model = None
                param_grid = {}
                scoring_metric = 'accuracy'

                if model_choice == "Linear Regression":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', LinearRegression())
                    ])
                    scoring_metric = 'r2'
                elif model_choice == "Logistic Regression":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000))
                    ])
                    scoring_metric = 'accuracy'
                    param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
                elif model_choice == "Random Forest Classifier":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())
                    ])
                    scoring_metric = 'accuracy'
                    param_grid = {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__max_depth': [None, 10, 20]
                    }
                elif model_choice == "Gradient Boosting Classifier":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', GradientBoostingClassifier())
                    ])
                    scoring_metric = 'accuracy'
                    param_grid = {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__learning_rate': [0.01, 0.1, 1]
                    }
                elif model_choice == "Support Vector Classifier (SVC)":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', SVC())
                    ])
                    scoring_metric = 'accuracy'
                    param_grid = {
                        'classifier__C': [0.01, 0.1, 1, 10, 100],
                        'classifier__kernel': ['linear', 'rbf']
                    }
                elif model_choice == "K-Nearest Neighbors (KNN)":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', KNeighborsClassifier())
                    ])
                    scoring_metric = 'accuracy'
                    param_grid = {
                        'classifier__n_neighbors': [3, 5, 7, 10]
                    }
                elif model_choice == "Decision Tree Classifier":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier())
                    ])
                    scoring_metric = 'accuracy'
                    param_grid = {
                        'classifier__max_depth': [None, 10, 20, 30]
                    }
                else:
                    flash('Invalid model choice!', 'danger')
                    return redirect(request.url)

                # Hyperparameter tuning using GridSearchCV
                if param_grid:
                    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring_metric)
                    grid_search.fit(X, y)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                else:
                    # No hyperparameters to tune
                    model.fit(X, y)
                    best_model = model
                    best_params = {}
                    best_score = cross_val_score(model, X, y, cv=5, scoring=scoring_metric).mean()

                # Train-test split and final evaluation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)

                # Score evaluation
                if scoring_metric == 'accuracy':
                    accuracy = accuracy_score(y_test, y_pred)
                else:
                    accuracy = r2_score(y_test, y_pred)

                # Save the best model and parameters in session for predictions
                session['best_model'] = best_model
                session['preprocessor'] = preprocessor
                session['model_params'] = best_params
                session['scoring_metric'] = scoring_metric

                flash(f"Model trained successfully! {scoring_metric.upper()} Score: {accuracy:.2f}", 'success')

                prediction_results = {
                    'best_score': best_score,
                    'best_params': best_params,
                    'accuracy': accuracy
                }

            except Exception as e:
                flash(f"An error occurred during model training: {e}", 'danger')
                return redirect(request.url)

        elif 'make_prediction' in request.form:
            try:
                # Ensure model parameters are in session
                if 'model_params' not in session:
                    flash('Please train a model first.', 'warning')
                    return redirect(request.url)

                model_params = session['model_params']
                scoring_metric = session['scoring_metric']

                # Reload the dataset and preprocess
                filepath = session.get('uploaded_file_path', None)
                if not filepath or not os.path.exists(filepath):
                    flash('No dataset uploaded!', 'danger')
                    return redirect(request.url)

                df = pd.read_csv(filepath)
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Recreate and fit the model with saved parameters
                model = None
                if model_choice == "Linear Regression":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('regressor', LinearRegression())
                    ])
                elif model_choice == "Logistic Regression":
                    model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000))
                    ])
                # Repeat for other models as needed...

                # Fit the model with the parameters
                model.fit(X, y)

                # Get user input for prediction
                user_input = {}
                feature_columns = request.form.getlist('feature_columns')
                for column in feature_columns:
                    value = request.form.get(column)
                    if value:
                        user_input[column] = value

                user_input_df = pd.DataFrame([user_input])

                # Transform and predict
                user_input_transformed = preprocessor.transform(user_input_df)
                prediction = model.predict(user_input_transformed)

                flash(f"Prediction: {prediction[0]}", 'success')
                prediction = prediction[0]

            except Exception as e:
                flash(f"An error occurred during prediction: {e}", 'danger')
                return redirect(request.url)

    # Load dataset to populate target columns and feature columns
    target_columns = []
    feature_columns = []
    model_options = [
        "Linear Regression",
        "Logistic Regression",
        "Random Forest Classifier",
        "Gradient Boosting Classifier",
        "Support Vector Classifier (SVC)",
        "K-Nearest Neighbors (KNN)",
        "Decision Tree Classifier"
    ]

    if 'uploaded_file_path' in session and session['uploaded_file_path']:
        try:
            df = pd.read_csv(session['uploaded_file_path'])
            target_columns = df.columns.tolist()
        except Exception as e:
            flash(f"Error loading dataset: {e}", 'danger')
            df = None

    return render_template(
        'predict.html',
        prediction_results=prediction_results,
        best_params=best_params,
        best_score=best_score,
        accuracy=accuracy,
        prediction=prediction,
        target_columns=target_columns,
        model_options=model_options,
        feature_columns=df.columns.tolist() if df is not None else []
    )

# Function to generate plots
def generate_plot(user_input, df):
    if df is None:
        return "No dataset uploaded for visualization."

    plot_message = ""
    if "heatmap" in user_input.lower():
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.savefig('static/heatmap.png')
        plt.close()
        plot_message = "Heatmap generated."
    elif "pairplot" in user_input.lower():
        sns.pairplot(df)
        plt.savefig('static/pairplot.png')
        plt.close()
        plot_message = "Pairplot generated."
    elif "histogram" in user_input.lower():
        # Expecting the user to specify the column name in some way
        # This requires additional handling; for simplicity, generate histograms for all numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure()
            df[col].hist(bins=30)
            plt.title(f'Histogram of {col}')
            plt.savefig(f'static/histogram_{col}.png')
            plt.close()
        plot_message = "Histograms generated for all numeric columns."
    else:
        plot_message = "Plot type not recognized."

    return plot_message

# Function to generate Sweetviz dashboard
def generate_dashboard(df):
    if df is None:
        return "No dataset uploaded for dashboard."
    
    report_path = 'static/sweetviz_report.html'
    
    # Generate the Sweetviz report
    try:
        report = sv.analyze(df)
        report.show_html(filepath=report_path, open_browser=False)
        # Check if the file was successfully created
        if os.path.exists(report_path):
            print(f"Sweetviz report saved at: {report_path}")
            return "Sweetviz dashboard generated."
        else:
            print(f"Failed to generate Sweetviz report at: {report_path}")
            return "Error: Failed to generate Sweetviz dashboard."
    except Exception as e:
        print(f"Exception during Sweetviz report generation: {e}")
        return "Error: Failed to generate Sweetviz dashboard."

if __name__ == '__main__':
    app.run(debug=True)
