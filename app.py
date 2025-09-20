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
    st.header("Menu Interativo")
    # --- MODIFICADO: Adicionada a nova página ao menu ---
    pagina_selecionada = st.radio(
        "Escolha uma seção:",
        ("Dashboard de Análise", "Módulo de Previsão", "Análise de Sentimentos")
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
# --- SEÇÃO 3: NOVA PÁGINA - ANÁLISE DE SENTIMENTOS ---
# ==============================================================================
elif pagina_selecionada == "Análise de Sentimentos":

    st.markdown("<h1 style='text-align: center; color: white;'>📜 Análise de Sentimentos</h1>", unsafe_allow_html=True)
    st.info("Esta seção exibe uma nuvem de palavras gerada a partir dos dados textuais.")

    try:
        # --- ATENÇÃO: Carregue aqui o seu DataFrame para esta análise ---
        # Substitua 'seu_arquivo_de_sentimentos.csv' pelo nome do arquivo correto.
        df_analise = pd.read_csv("seu_arquivo_de_sentimentos.csv")
        
        # Verifique se a coluna necessária existe
        if 'letra_processada' not in df_analise.columns:
            st.error("Erro: A coluna 'letra_processada' não foi encontrada no arquivo de dados.")
        else:
            # --- SEU CÓDIGO DA NUVEM DE PALAVRAS (ADAPTADO PARA STREAMLIT) ---
            st.subheader("Nuvem de Palavras Mais Frequentes")
            
            # Defina sua lista de stopwords customizadas (se tiver)
            stoplist_custom = ["aqui", "outra", "palavra"] # Exemplo, ajuste conforme sua necessidade
            
            texto_completo = " ".join(df_analise['letra_processada'].dropna())

            with st.spinner("Gerando nuvem de palavras..."):
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="black",
                    colormap="Dark2",
                    stopwords=nlp.Defaults.stop_words.union(stoplist_custom),
                    collocations=False,
                    min_font_size=10,
                    max_words=200
                ).generate(texto_completo)

                # Para exibir no Streamlit, criamos uma figura e um eixo com matplotlib
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.style.use("dark_background") # Define o estilo do gráfico
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")

                # Comando para mostrar a figura do matplotlib no Streamlit
                st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Para gerar a nuvem de palavras, por favor, adicione o arquivo de dados (ex: 'seu_arquivo_de_sentimentos.csv') ao seu repositório.")
    
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")

    # Rodapé
    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")