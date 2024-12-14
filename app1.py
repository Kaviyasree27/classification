import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model and LabelEncoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the dataset for feature options
df = pd.read_csv('mushrooms.csv')

# All features used in training
all_features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]

# Selected features for classification
selected_features = [
    'bruises', 'gill-size', 'gill-spacing', 'gill-color',
    'stalk-surface-below-ring', 'veil-color', 'ring-type',
    'population', 'habitat'
]

# Streamlit Interface
st.markdown('<h1 style="text-align: center; color: #4CAF50;">🍄 Mushroom Classification App</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #FF5722;">Predict whether a mushroom is Edible or Poisonous!</h3>', unsafe_allow_html=True)
st.write("Fill in the following details about the mushroom to get a prediction:")

# Create input fields for selected features
st.sidebar.header('Input Features')
st.sidebar.markdown('<p style="color: #3F51B5;">Please select values for the following features:</p>', unsafe_allow_html=True)
user_inputs = {}
for feature in selected_features:
    user_inputs[feature] = st.sidebar.selectbox(
        label=f"{feature.replace('-', ' ').title()}", 
        options=df[feature].unique(),
        help=f"Select the value for {feature}"
    )

# Create a complete input data array with default values
input_data = []
for feature in all_features:
    if feature in selected_features:
        if user_inputs[feature] in label_encoder.classes_:
            input_data.append(label_encoder.transform([user_inputs[feature]])[0])
        else:
            # Handle unseen labels
            input_data.append(0)  # Default value for unseen labels
    else:
        # Use default values for other features
        input_data.append(0)  # Default value

# Convert the inputs into a DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=all_features)

# Add a stylish prediction button
if st.button('🔍 Classify Mushroom'):
    # Perform prediction
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0]  # Get probabilities for both classes

    # Extract probabilities for edible and poisonous
    edible_prob = prediction_prob[0]  # Probability for edible (class 0)
    poisonous_prob = prediction_prob[1]  # Probability for poisonous (class 1)

    # Display probabilities and result
    if prediction == 1:  # Poisonous
        st.markdown('<p style="color: #D32F2F; font-size: 18px;">🚨 <b>Prediction: Poisonous</b> (Probability: {:.2%})</p>'.format(poisonous_prob), unsafe_allow_html=True)
        st.markdown('<p style="color: #F44336;">⚠️ <b>Warning!</b> This mushroom is highly poisonous. Avoid consumption!</p>', unsafe_allow_html=True)
    elif prediction == 0:  # Edible
        st.markdown('<p style="color: #4CAF50; font-size: 18px;">✅ <b>Prediction: Edible</b> (Probability: {:.2%})</p>'.format(edible_prob), unsafe_allow_html=True)
        st.markdown('<p style="color: #8BC34A;">🌟 This mushroom is safe to eat. Enjoy!</p>', unsafe_allow_html=True)

    # Show detailed probabilities
    st.markdown("### Prediction Probabilities:")
    st.progress(poisonous_prob)
    st.markdown('<p style="color: #3F51B5;">**Edible:** {:.2%}</p>'.format(edible_prob), unsafe_allow_html=True)
    st.markdown('<p style="color: #D32F2F;">**Poisonous:** {:.2%}</p>'.format(poisonous_prob), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #607D8B;">Developed with ❤️ using <a href="https://streamlit.io/" style="color: #FF5722; text-decoration: none;">Streamlit</a> - KAVIYA SREE R.S</p>',
    unsafe_allow_html=True
)
