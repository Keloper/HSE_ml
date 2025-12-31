import pickle
import streamlit as st

@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


