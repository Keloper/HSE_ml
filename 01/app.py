import pickle
import streamlit as st 
import os 
from sklearn.linear_model import Ridge

st.set_page_config(
    page_title="Price prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)




# Загрузка модели 

@st.cache_resource
def load_model():
    model = Ridge()
    model.load_model('models/Ridge_model.pkl')
    feature_names = model.feature_names_
    return model, feature_names


@st.cache_data  # Кэшируем загруженные данные
def prepare_features(df, feature_names):
    """Приводим данные к формату обучения модели"""
    df_proc = df.copy()
    # Преобразуем категориальные признаки в строки (как при обучении)
    for col in feature_names:
        if col in df_proc.columns:
            if df_proc[col].dtype in ('object', 'bool'):
                df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]