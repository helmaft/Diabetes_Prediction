import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set the random seed
random_seed = 42

# Load the dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Drop duplicates
df = df.drop_duplicates()

# Apply data preprocessing steps
def preprocess_data(df):
    # Map smoking status categories
    df['smoking_history'] = df['smoking_history'].map({
        'never': 'non-smoker',
        'No Info': 'non-smoker',
        'current': 'current',
        'ever': 'past_smoker',
        'former': 'past_smoker',
        'not current': 'past_smoker'
    })

    # Label encode categorical variables
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])

    # Scale numerical variables
    scaler = StandardScaler()
    columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df

df = preprocess_data(df)

# Train the models
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

logreg_model = LogisticRegression(random_state=random_seed)
logreg_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=random_seed)
rf_model.fit(X_train, y_train)

# Define the Streamlit app
def main():
    st.title("Diabetes Prediction App")

    st.markdown("## Data Overview")
    st.write(df.head())

    st.markdown("## Data Visualization")
    st.markdown("### Gender Distribution")
    fig_gender = px.bar(df['gender'].value_counts(), x=df['gender'].value_counts().index, y=df['gender'].value_counts().values,
                        labels={'x': 'Gender', 'y': 'Count'})
    st.plotly_chart(fig_gender)

    st.markdown("### Age Distribution")
    fig_age = px.histogram(df, x='age', nbins=50)
    st.plotly_chart(fig_age)

    st.markdown("### Smoking History")
    fig_smoking = px.pie(df, names='smoking_history')
    st.plotly_chart(fig_smoking)

    st.markdown("## Model Comparison")
    models = ['Random Forest', 'Logistic Regression']
    accuracies = [rf_model.score(X_test, y_test), logreg_model.score(X_test, y_test)]
    fig_accuracies = px.bar(x=models, y=accuracies, labels={'x': 'Models', 'y': 'Accuracy'})
    st.plotly_chart(fig_accuracies)

if __name__ == '__main__':
    main()
