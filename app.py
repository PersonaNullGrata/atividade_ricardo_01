import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Naïve Bayes Apps", page_icon="🧠", layout="wide")

# Custom CSS para melhorar o visual
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #125b8a;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧠 Navegação")
st.sidebar.markdown("Selecione qual atividade deseja explorar:")
page = st.sidebar.radio("", [
    "1. Atividade 1 (Básico)", 
    "2. Atividade 2 (Intermediário)", 
    "3. Atividade 3 (NLP)"
])

if page == "1. Atividade 1 (Básico)":
    st.title("Gaussian Naïve Bayes")
    st.write("Modelo treinado com **dados sintéticos contínuos** para prever uma classificação binária.")
    st.markdown("---")
    
    try:
        model = joblib.load('modelo_nb1.pkl')
        
        col_pred, col_metrics = st.columns([1, 1])
        
        with col_pred:
            st.subheader("💡 Fazer Predição")
            st.info("Insira os valores contínuos para as 4 features sintéticas.")
            f1 = st.number_input("Feature 1", value=0.0, step=0.1)
            f2 = st.number_input("Feature 2", value=0.0, step=0.1)
            f3 = st.number_input("Feature 3", value=0.0, step=0.1)
            f4 = st.number_input("Feature 4", value=0.0, step=0.1)
                
            if st.button("🔮 Classificar Dados", key="btn_nb1"):
                pred = model.predict([[f1, f2, f3, f4]])
                st.success(f"A classe predita para as features fornecidas é: **{pred[0]}**")
                
        with col_metrics:
            st.subheader("📊 Métricas de Avaliação")
            try:
                img_cm = Image.open('cm_nb1.png')
                st.image(img_cm, caption='Matriz de Confusão', use_container_width=True)
            except FileNotFoundError:
                st.warning("Matriz de confusão não encontrada.")
            
            try:
                img_roc = Image.open('roc_nb1.png')
                st.image(img_roc, caption='Curva ROC', use_container_width=True)
            except FileNotFoundError:
                st.warning("Curva ROC não encontrada.")
                
    except FileNotFoundError:
        st.error("⚠️ O modelo não foi encontrado. Por favor, certifique-se de executar `python naive-bayes-1.py` primeiro para gerar o arquivo `.pkl`.")


elif page == "2. Atividade 2 (Intermediário)":
    st.title("Categorical Naïve Bayes")
    st.write("Modelo treinado para prever o **perfil de risco de crédito** baseado em dados puramente categóricos.")
    st.markdown("---")
    
    try:
        pipeline = joblib.load('modelo_nb2.pkl')
        
        col_pred, col_metrics = st.columns([1, 1])
        
        with col_pred:
            st.subheader("💡 Fazer Predição")
            st.info("Preencha o perfil do cliente abaixo.")
            escolaridade = st.selectbox("Escolaridade", ['Medio', 'Superior', 'Pos-Graduacao'])
            renda = st.selectbox("Faixa de Renda", ['Baixa', 'Media', 'Alta'])
            historico = st.selectbox("Histórico de Crédito", ['Ruim', 'Bom', 'Excelente'])
                
            if st.button("🔮 Avaliar Risco", key="btn_nb2"):
                df_input = pd.DataFrame({
                    'escolaridade': [escolaridade], 
                    'renda_faixa': [renda], 
                    'historico_credito': [historico]
                })
                pred = pipeline.predict(df_input)
                prob = pipeline.predict_proba(df_input)
                confianca = np.max(prob) * 100
                
                if pred[0] == 'Baixo':
                    st.success(f"O perfil de risco predito é: **{pred[0]}** (Confiança: {confianca:.1f}%)")
                else:
                    st.error(f"O perfil de risco predito é: **{pred[0]}** (Confiança: {confianca:.1f}%)")
                
        with col_metrics:
            st.subheader("📊 Métricas de Avaliação")
            try:
                img_cm = Image.open('cm_nb2.png')
                st.image(img_cm, caption='Matriz de Confusão', use_container_width=True)
            except FileNotFoundError:
                st.warning("Matriz de confusão não encontrada.")
            
            try:
                img_roc = Image.open('roc_nb2.png')
                st.image(img_roc, caption='Curva ROC', use_container_width=True)
            except FileNotFoundError:
                st.warning("Curva ROC não encontrada.")
                
    except FileNotFoundError:
        st.error("⚠️ O modelo não foi encontrado. Por favor, certifique-se de executar `python naive-bayes-2.py` primeiro para gerar o arquivo `.pkl`.")


elif page == "3. Atividade 3 (NLP)":
    st.title("Multinomial Naïve Bayes (NLP)")
    st.write("Modelo de Processamento de Linguagem Natural (NLP) treinado para classificar o **sentimento de avaliações de produtos**.")
    st.markdown("---")
    
    try:
        pipeline_nlp = joblib.load('modelo_nb3.pkl')
        
        col_pred, col_metrics = st.columns([1, 1])
        
        with col_pred:
            st.subheader("💡 Análise de Sentimento")
            st.info("Digite uma avaliação de produto (ex: 'O produto chegou quebrado' ou 'Muito bom e rápido').")
            texto = st.text_area("Avaliação do Produto", height=150)
            
            if st.button("🔮 Classificar Texto", key="btn_nb3"):
                if texto.strip():
                    pred = pipeline_nlp.predict([texto])
                    prob = pipeline_nlp.predict_proba([texto])
                    classe = "Positivo" if pred[0] == 1 else "Negativo"
                    confianca = np.max(prob) * 100
                    
                    if pred[0] == 1:
                        st.success(f"Sentimento **{classe}** com {confianca:.1f}% de confiança. 😃")
                    else:
                        st.error(f"Sentimento **{classe}** com {confianca:.1f}% de confiança. 😡")
                else:
                    st.warning("Por favor, digite algum texto para ser analisado.")
                    
        with col_metrics:
            st.subheader("📊 Métricas de Avaliação")
            try:
                img_cm = Image.open('cm_nb3.png')
                st.image(img_cm, caption='Matriz de Confusão', use_container_width=True)
            except FileNotFoundError:
                st.warning("Matriz de confusão não encontrada.")
            
            try:
                img_roc = Image.open('roc_nb3.png')
                st.image(img_roc, caption='Curva ROC', use_container_width=True)
            except FileNotFoundError:
                st.warning("Curva ROC não encontrada.")

    except FileNotFoundError:
        st.error("⚠️ O modelo não foi encontrado. Por favor, certifique-se de executar `python pln-naive-bayes-3.py` primeiro para gerar o arquivo `.pkl`.")
