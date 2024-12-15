

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


model_path = 'final_model.pkl'
scaler_path = 'scaler.pkl'
encoder_path = 'target_encoder.pkl'
label_path = 'label_encoder.pkl'

final_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
target_encoder = joblib.load(encoder_path)
label_encoder = joblib.load(label_path)

include_variables = [
    'Attorney/Representative',
    'Average Weekly Wage',
    'Birth Year',
    'Carrier Name',
    'Carrier Type',
    'IME-4 Count',
    'Industry Code',
    'Medical Fee Region',
    'WCIO Cause of Injury Code',
    'WCIO Nature of Injury Code',
    'WCIO Part Of Body Code',
    'Zip Code',
    'Agreement Reached',
    'C-3 Received',
    'First Hearing Happened',
    'Time to Assembly'
]


left_col_features = [
    'Average Weekly Wage',
    'Birth Year',
    'Carrier Name',
    'Carrier Type',
    'Industry Code',
    'Zip Code',
    'IME-4 Count',
    'Medical Fee Region',
]

right_col_features = [
    'Attorney/Representative',
    'Time to Assembly',
    'WCIO Cause of Injury Code',
    'WCIO Nature of Injury Code',
    'WCIO Part Of Body Code',
    'Agreement Reached',
    'C-3 Received',
    'First Hearing Happened'
]

st.title("Model Prediction App")
st.write("Welcome to the prediction app! You can make predictions using the trained model.")

st.subheader("Enter the Values for Prediction")
input_data = {}

col1, col2 = st.columns(2)

with col1:
    for feature in left_col_features:
        if feature in ['Zip Code']:
            input_data[feature] = st.text_input(feature, "").zfill(5)
        elif feature in ['Birth Year']:
            input_data[feature] = st.text_input(feature, "").zfill(4)
        elif feature in ['Carrier Name', 'Carrier Type', 'Medical Fee Region']:
            input_data[feature] = st.text_input(feature, "").upper()
        elif feature in ['Average Weekly Wage', 'IME-4 Count']:
            input_data[feature] = st.number_input(feature, value=0.0)
        else:
            input_data[feature] = st.number_input(feature, value=0)

with col2:
    for feature in right_col_features:
        if feature in ['Attorney/Representative', 'Agreement Reached', 'C-3 Received', 'First Hearing Happened']:
            input_data[feature] = st.number_input(f"{feature} (0 for No and 1 for Yes)", min_value=0, max_value=1, step=1)
        else:
            input_data[feature] = st.number_input(feature, value=0)

            
if st.button("Predict"):
    with st.spinner("Making prediction..."):
        input_df = pd.DataFrame([input_data], columns=include_variables)

        encoder_features = target_encoder.feature_names_in_
        
        for feature in encoder_features:
            if feature not in input_df.columns:
                input_df[feature] = ""

        cols_to_enc = [
            'Carrier Name', 'Carrier Type', 'County of Injury', 'Industry Code', 
            'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code', 
            'WCIO Part Of Body Code', 'District Name', 
            'Gender', 'Medical Fee Region', 'Zip Code', 'Injury Day of Week'
        ] 
        
        input_df[cols_to_enc] = target_encoder.transform(input_df[cols_to_enc])

        scaler_features = scaler.feature_names_in_

        for feature in scaler_features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[scaler_features]
        
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=scaler.feature_names_in_)
        
        input_df_for_pred = scaled_input_df[include_variables]
        
        encoded_prediction = final_model.predict(input_df_for_pred)
        prediction = label_encoder.inverse_transform(encoded_prediction)

        if hasattr(final_model, "predict_proba"):
            probabilities = final_model.predict_proba(input_df_for_pred)
            confidence_score = probabilities.max(axis=1)[0]

        st.success("Prediction Complete!")
        st.write(f"Prediction: {prediction[0]}")  

        if hasattr(final_model, "predict_proba"):
            st.metric(label="Confidence Score", value=f"{confidence_score * 100:.2f}%")

        rf_model = final_model.named_estimators_['rf']
        rf_feature_importances = rf_model.feature_importances_

        feature_importances_df = pd.DataFrame({
            'Feature': input_df_for_pred.columns,
            'Importance': rf_feature_importances
        })

        # Sorting the feature importances
        feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

        # Display the top feature importances
        st.write("Top Feature Importances from RandomForestClassifier:")
        fig, ax = plt.subplots()
        ax.bar(feature_importances_df['Feature'], feature_importances_df['Importance'])
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        plt.xticks(rotation=45, ha='right')  # Rotate labels
        st.pyplot(fig)
