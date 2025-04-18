import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
import numpy as np

def main():
    st.set_page_config(page_title="Chat with ME!", layout="wide")
    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Go to", ["Chatbot", "Model Predictions"])
    groq_api_key = os.environ.get('GROQ_API_KEY')

    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable is not set.")
        return

    if page == "Chatbot":
        chatbot_page(groq_api_key)
    elif page == "Model Predictions":
        model_predictions_page(groq_api_key)

def chatbot_page(groq_api_key):
    # Display the Groq logo
    spacer, col = st.columns([5, 1])
    with col:
        st.image('logo.png')

    st.title("Chat with ME!")
    st.write("Hello! I'm your friendly Neighbourhood chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV)", type=["csv"])

    df = None  # Initialize df outside the if block
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Dataset Uploaded Successfully!")
            st.sidebar.write(df.head())  # Display the first few rows of the dataset
        except Exception as e:
            st.sidebar.error(f"Error reading the CSV file: {e}")

    user_question = st.text_input("Ask a question:")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    if user_question:
        if "heatmap" in user_question.lower() or "histogram" in user_question.lower() or "pairplot" in user_question.lower() or "barplot" in user_question.lower():
            st.write("Generating visualization...")

            if df is not None:  # Ensure df is not None
                if "heatmap" in user_question.lower():
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
                    st.pyplot(plt.gcf())
                    plt.clf()

                elif "pairplot" in user_question.lower():
                    sns.pairplot(df)
                    st.pyplot(plt.gcf())
                    plt.clf()

                elif "histogram" in user_question.lower():
                    column = st.text_input("Enter the column name for histogram:")
                    if column in df.columns:
                        plt.figure(figsize=(10, 6))
                        df[column].hist(bins=30)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.write("Column not found in the dataset.")

                elif "barplot" in user_question.lower():
                    column = st.text_input("Enter the column name for barplot:")
                    if column in df.columns:
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts().values)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    else:
                        st.write("Column not found in the dataset.")

            else:
                st.write("Please upload a dataset first.")

            response = "Here's the visualization you requested."
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )

            dataset_info_message = "No dataset uploaded."
            if df is not None:
                dataset_info = df.describe().to_string()
                dataset_info_message = f"The dataset has been uploaded. Here are some insights:\n{dataset_info}"

            response = conversation.predict(human_input=f"{user_question}\n\n{dataset_info_message}")

        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

def model_predictions_page(groq_api_key):
    st.title("Model Predictions")

    uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Uploaded Successfully!")
            st.write(df.head())

            target_column = st.selectbox("Select the target column for prediction:", df.columns)

            if target_column:
                model_options = [
                    "Linear Regression",
                    "Logistic Regression",
                    "Random Forest Classifier",
                    "Gradient Boosting Classifier",
                    "Support Vector Classifier (SVC)",
                    "K-Nearest Neighbors (KNN)",
                    "Decision Tree Classifier"
                ]

                model_choice = st.selectbox("Choose a model to apply:", model_options)

                if st.button("Train Model"):
                    X = df.drop(columns=[target_column])
                    y = df[target_column]

                    categorical_cols = X.select_dtypes(include=['object']).columns

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                            ('num', SimpleImputer(strategy='mean'), X.columns.difference(categorical_cols))
                        ],
                        remainder='passthrough'
                    )

                    model = None
                    param_grid = {}

                    if model_choice == "Linear Regression":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('regressor', LinearRegression())])
                        scoring_metric = 'r2'
                        model_step_name = 'regressor'  # Define step name for regression
                    elif model_choice == "Logistic Regression":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', LogisticRegression())])
                        scoring_metric = 'accuracy'
                        param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
                        model_step_name = 'classifier'  # Define step name for classifier
                    elif model_choice == "Random Forest Classifier":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', RandomForestClassifier())])
                        scoring_metric = 'accuracy'
                        param_grid = {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 10, 20]}
                        model_step_name = 'classifier'
                    elif model_choice == "Gradient Boosting Classifier":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', GradientBoostingClassifier())])
                        scoring_metric = 'accuracy'
                        param_grid = {'classifier__n_estimators': [50, 100, 200], 'classifier__learning_rate': [0.01, 0.1, 1]}
                        model_step_name = 'classifier'
                    elif model_choice == "Support Vector Classifier (SVC)":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', SVC())])
                        scoring_metric = 'accuracy'
                        param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__kernel': ['linear', 'rbf']}
                        model_step_name = 'classifier'
                    elif model_choice == "K-Nearest Neighbors (KNN)":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', KNeighborsClassifier())])
                        scoring_metric = 'accuracy'
                        param_grid = {'classifier__n_neighbors': [3, 5, 7, 10]}
                        model_step_name = 'classifier'
                    elif model_choice == "Decision Tree Classifier":
                        model = Pipeline(steps=[('preprocessor', preprocessor),
                                                ('classifier', DecisionTreeClassifier())])
                        scoring_metric = 'accuracy'
                        param_grid = {'classifier__max_depth': [None, 10, 20, 30]}
                        model_step_name = 'classifier'

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

                    st.write(f"Best Score (cross-validation): {best_score:.2f}")
                    st.write(f"Best Parameters: {best_params}")

                    # Train-test split and evaluation
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    best_model.fit(X_train, y_train)
                    y_pred = best_model.predict(X_test)

                    if scoring_metric == 'accuracy':
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"Model Accuracy: {accuracy:.2f}")
                    else:
                        accuracy = r2_score(y_test, y_pred)
                        st.write(f"Model R^2 Score: {accuracy:.2f}")

                    st.write("Enter values for the independent variables:")

                    user_input = {}
                    for column in X.columns:
                        if column in categorical_cols:
                            user_input[column] = st.selectbox(f"Select value for {column}:", df[column].unique())
                        else:
                            user_input[column] = st.number_input(f"Enter value for {column}:", value=float(df[column].mean()))

                    user_input_df = pd.DataFrame([user_input])
                    st.write("User Input Data:")
                    st.write(user_input_df)

                    user_input_transformed = best_model.named_steps['preprocessor'].transform(user_input_df)

                    prediction = best_model.named_steps[model_step_name].predict(user_input_transformed)

                    st.write("Prediction based on user input:")
                    st.write(prediction[0])

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()