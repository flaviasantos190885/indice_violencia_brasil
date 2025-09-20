# --- IMPORTS NOVOS E ANTIGOS ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import warnings
# --- ADICIONADO: Imports para a nuvem de palavras ---
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy

# --- CONFIGURAÇÃO DA PÁGINA E AVISOS ---
st.set_page_config(layout="wide", page_title="Análise de Violência no Brasil")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- ADICIONADO: Carregar modelo de linguagem para stopwords ---
# (Isso pode demorar um pouco na primeira vez que o app rodar)
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    st.info("Baixando modelo de linguagem para processamento de texto...")
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load('pt_core_news_sm')


# --- FUNÇÃO DE CACHE PARA CARREGAR OS ATIVOS DE PREVISÃO ---
@st.cache_resource
def carregar_ativos_previsao():
    """Carrega o modelo, o pré-processador e o normalizador salvos."""
    try:
        model = load_model('melhor_modelo_multivariado.keras')
        preprocessor = joblib.load('preprocessor.joblib')
        y_scaler = joblib.load('y_scaler.joblib')
        return model, preprocessor, y_scaler
    except FileNotFoundError:
        return None, None, None

# --- CARREGAMENTO INICIAL DE DADOS ---
try:
    df_completo = pd.read_csv("Dados_2015_2024.csv")
    df_completo['data_referencia'] = pd.to_datetime(df_completo['data_referencia'], errors='coerce')
    df_completo['Ano'] = df_completo['data_referencia'].dt.year # Garante que a coluna 'Ano' existe
except FileNotFoundError:
    st.error("Erro: O arquivo 'Dados_2015_2024.csv' não foi encontrado.")
    st.stop()

# --- BARRA LATERAL DE NAVEGAÇÃO ---
with st.sidebar:
    # --- CÓDIGO CSS PARA ADICIONAR ESPAÇAMENTO ---
    st.markdown("""
    <style>
        div[role="radiogroup"] > div {
            margin-bottom: 15px; /* Aumenta o espaço abaixo de cada item */
        }
    </style>
    """, unsafe_allow_html=True)

    st.header("Menu Interativo")
    pagina_selecionada = st.radio(
        "Escolha uma seção:",
        ("Dashboard de Análise", "Módulo de Previsão", "Análise de Palavras")
    )
    st.markdown("---")
    st.info("Este painel oferece uma análise visual dos dados de violência e um módulo para estimativas futuras.")

# ==============================================================================
# --- SEÇÃO 1: DASHBOARD DE ANÁLISE ---
# ==============================================================================
if pagina_selecionada == "Dashboard de Análise":
    # (Todo o seu código do dashboard continua aqui, sem alterações)
    # ...
    st.markdown("<h1 style='text-align: center; font-size: 40px; color: white'>📊 Dados da Violência no Brasil</h1>", unsafe_allow_html=True)
    # ... (o resto do seu código gigante do dashboard) ...
    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")


# ==============================================================================
# --- SEÇÃO 2: MÓDULO DE PREVISÃO ---
# ==============================================================================
elif pagina_selecionada == "Módulo de Previsão":
    # (Todo o seu código do módulo de previsão continua aqui, sem alterações)
    # ...
    st.markdown("<h1 style='text-align: center; color: white;'>🧠 Módulo de Previsão Anual</h1>", unsafe_allow_html=True)
    # ... (o resto do seu código gigante da previsão) ...
    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")


# ==============================================================================
# --- SEÇÃO 3: NOVA PÁGINA - ANÁLISE DE PALAVRAS ---
# ==============================================================================
elif pagina_selecionada == "Análise de Palavras":

    # --- FUNÇÃO AUXILIAR PARA GERAR NUVEM DE PALAVRAS ---
    # Colocamos a lógica dentro de uma função para poder reutilizá-la facilmente
    def gerar_nuvem_de_palavras(dataframe, nome_coluna, titulo):
        """
        Gera e exibe uma nuvem de palavras para uma coluna específica de um DataFrame.
        """
        st.subheader(titulo)
        
        # Garante que estamos pegando apenas textos, removendo valores nulos e convertendo para string
        texto_completo = " ".join(dataframe[nome_coluna].dropna().astype(str))

        # Verifica se há texto para processar
        if not texto_completo.strip():
            st.warning(f"Não há dados suficientes na coluna '{nome_coluna}' para gerar a nuvem de palavras.")
            return # Sai da função se não houver texto

        with st.spinner(f"Gerando nuvem para '{nome_coluna}'..."):
            
            # --- MODIFICADO: Removido o parâmetro 'max_words' para incluir todas as palavras ---
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="black",
                colormap="Dark2",
                stopwords=nlp.Defaults.stop_words, # Você pode adicionar sua lista customizada aqui se precisar
                collocations=False,
                min_font_size=10
            ).generate(texto_completo)

            # Para exibir no Streamlit, criamos uma figura com matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use("dark_background")
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")

            # Comando para mostrar a figura do matplotlib no Streamlit
            st.pyplot(fig)

    # --- TÍTULO PRINCIPAL DA PÁGINA ---
    st.markdown("<h1 style='text-align: center; color: white;'>📜 Análise de Palavras-Chave</h1>", unsafe_allow_html=True)
    st.info("Esta seção exibe as palavras mais frequentes nas colunas 'evento' e 'arma' dos registros.")

    try:
        df_analise = df_completo.copy()
        
        # --- CHAMADA 1: GERAR A NUVEM PARA A COLUNA 'EVENTO' ---
        gerar_nuvem_de_palavras(df_analise, 'evento', 'Nuvem de Palavras por Tipo de Evento')
        
        st.markdown("---") # Adiciona uma linha para separar os gráficos
        
        # --- CHAMADA 2: GERAR A NUVEM PARA A COLUNA 'ARMA' ---
        gerar_nuvem_de_palavras(df_analise, 'arma', 'Nuvem de Palavras por Tipo de Arma')

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao gerar as nuvens de palavras: {e}")

    # Rodapé
    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")