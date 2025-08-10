import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import warnings

# --- CONFIGURAÇÃO DA PÁGINA E AVISOS ---
st.set_page_config(layout="wide", page_title="Análise de Violência no Brasil")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- FUNÇÃO DE CACHE PARA CARREGAR OS ATIVOS DE PREVISÃO ---
# @st.cache_resource garante que o modelo pesado e os arquivos sejam carregados apenas uma vez.
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
# Carrega o dataset para o dashboard e para a lógica de previsão
try:
    df_completo = pd.read_csv("Dados_2015_2024.csv")
    df_completo['data_referencia'] = pd.to_datetime(df_completo['data_referencia'])
except FileNotFoundError:
    st.error("Erro: O arquivo 'Dados_2015_2024.csv' não foi encontrado. Por favor, coloque-o na mesma pasta.")
    st.stop() # Interrompe a execução se o arquivo principal não for encontrado

# --- BARRA LATERAL DE NAVEGAÇÃO ---
with st.sidebar:
    st.header("Navegação")
    pagina_selecionada = st.radio(
        "Escolha uma seção:",
        ("Dashboard de Análise", "Módulo de Previsão")
    )
    st.markdown("---")
    st.info("Este painel oferece uma análise visual dos dados de violência e um módulo para estimativas futuras.")


# ==============================================================================
# --- SEÇÃO 1: DASHBOARD DE ANÁLISE (SEU CÓDIGO ORIGINAL) ---
# ==============================================================================
if pagina_selecionada == "Dashboard de Análise":
    
    st.markdown("<h1 style='text-align: center; color: white;'>📊 Dashboard da Violência no Brasil</h1>", unsafe_allow_html=True)
    st.markdown("---")

    df = df_completo.copy()
    df['Ano'] = df['data_referencia'].dt.year
    df['Mes'] = df['data_referencia'].dt.month_name()
    meses_pt = {'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril','May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto','September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'}
    df['Mes'] = df['Mes'].map(meses_pt)
    
    # Seus filtros e gráficos do dashboard original...
    anos = sorted(df['Ano'].unique())
    todos_estados = sorted(df['uf'].unique())
    eventos = sorted(df['evento'].unique())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ano_selecionado = st.selectbox("Selecione o Ano", anos)
    with col2:
        estado_selecionado = st.multiselect("Selecione os Estados", options=todos_estados, placeholder="Todos os Estados")
    with col3:
        evento_input = st.selectbox("Tipo de Evento", ["Todos"] + eventos)

    if not estado_selecionado:
        estados_filtrados = todos_estados
    else:
        estados_filtrados = estado_selecionado

    if len(estados_filtrados) == 1:
        cidades = df[df['uf'] == estados_filtrados[0]]['municipio'].sort_values().unique()
        cidade_input = st.selectbox("Selecione a Cidade", ["Todas"] + list(cidades), index=0)
    else:
        st.selectbox("Selecione a Cidade", ["Todas"], index=0, disabled=True)
        cidade_input = "Todas"

    df_filtrado = df[df['Ano'] == ano_selecionado]
    df_filtrado = df_filtrado[df_filtrado['uf'].isin(estados_filtrados)]
    if cidade_input != "Todas":
        df_filtrado = df_filtrado[df_filtrado['municipio'] == cidade_input]
    if evento_input != "Todos":
        df_filtrado = df_filtrado[df_filtrado['evento'] == evento_input]
    
    # Seus gráficos (Barra, Linha, Pizza) e Tabela...
    st.markdown("### Total de Vítimas por Estado")
    df_barra = df_filtrado.groupby('uf')['total_vitima'].sum().reset_index()
    fig_barra = px.bar(df_barra, x='uf', y='total_vitima', text_auto=True, labels={'uf': 'Estado', 'total_vitima': 'Total de Vítimas'}, color='uf')
    st.plotly_chart(fig_barra, use_container_width=True)

    st.markdown("### Evolução Mensal dos Casos por Estado")
    df_linha = df_filtrado.groupby(['uf', 'Mes'])['total_vitima'].sum().reset_index()
    ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho','Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    df_linha['Mes'] = pd.Categorical(df_linha['Mes'], categories=ordem_meses, ordered=True)
    df_linha = df_linha.sort_values(['uf', 'Mes'])
    fig_linha = px.line(df_linha,x='Mes',y='total_vitima',color='uf',markers=True,labels={'Mes': 'Mês','total_vitima': 'Total de Vítimas','uf': 'Estado'})
    st.plotly_chart(fig_linha, use_container_width=True)

    st.markdown("### Dados Filtrados")
    st.dataframe(df_filtrado.drop(columns=['Ano', 'Mes']))


# ==============================================================================
# --- SEÇÃO 2: MÓDULO DE PREVISÃO ---
# ==============================================================================
elif pagina_selecionada == "Módulo de Previsão":
    
    st.markdown("<h1 style='text-align: center; color: white;'>🧠 Módulo de Previsão Anual</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Use este módulo para gerar uma estimativa de vítimas para um ano futuro, com base no modelo treinado com dados históricos e em filtros opcionais.")

    # Carrega o modelo e os pré-processadores
    model, preprocessor, y_scaler = carregar_ativos_previsao()
    
    if not model:
        st.error("Arquivos de modelo não encontrados! Certifique-se de que 'melhor_modelo_multivariado.keras', 'preprocessor.joblib' e 'y_scaler.joblib' estão na pasta.")
        st.stop()
        
    # Botão para abrir o popup (dialog) de previsão
    if st.button("🚀 Iniciar Nova Previsão", type="primary"):
        with st.dialog("Parâmetros da Previsão", width="large"):
            st.markdown("#### Preencha os campos para gerar a estimativa:")
            
            # INPUTS DENTRO DO POPUP
            ano_desejado = st.number_input("Digite o ANO para a previsão (Obrigatório)", min_value=df_completo['Ano'].max() + 1, value=df_completo['Ano'].max() + 1, step=1)
            
            col_filtros1, col_filtros2 = st.columns(2)
            with col_filtros1:
                uf_selecionada = st.selectbox("Filtrar por UF (Opcional)", ["Todos"] + sorted(df_completo['uf'].unique()))
                arma_selecionada = st.selectbox("Filtrar por Arma (Opcional)", ["Todos"] + sorted(df_completo['arma'].unique()))
            with col_filtros2:
                evento_selecionado = st.selectbox("Filtrar por Evento (Opcional)", ["Todos"] + sorted(df_completo['evento'].unique()))
                faixa_selecionada = st.selectbox("Filtrar por Faixa Etária (Opcional)", ["Todos"] + sorted(df_completo['faixa_etaria'].unique()))

            # BOTÃO PARA CALCULAR DENTRO DO POPUP
            if st.button("Calcular Estimativa"):
                df_filtrado_pred = df_completo.copy()
                
                # Aplica filtros opcionais
                if uf_selecionada != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['uf'] == uf_selecionada]
                if evento_selecionado != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['evento'] == evento_selecionado]
                if arma_selecionada != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['arma'] == arma_selecionada]
                if faixa_selecionada != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['faixa_etaria'] == faixa_selecionada]

                # Lógica de previsão (mesma do script anterior)
                janela = 10
                if len(df_filtrado_pred) < janela:
                    st.error(f"Dados históricos insuficientes ({len(df_filtrado_pred)} eventos) para o cenário. Tente filtros menos específicos.")
                else:
                    with st.spinner("Calculando... O modelo está processando os dados."):
                        num_anos_historico = df_filtrado_pred['Ano'].nunique()
                        media_eventos_ano = len(df_filtrado_pred) / num_anos_historico if num_anos_historico > 0 else 0
                        
                        sequencia_base = df_filtrado_pred.tail(janela - 1).copy()
                        evento_futuro_template = df_filtrado_pred.tail(1).copy()
                        evento_futuro_template['Ano'] = ano_desejado
                        
                        sequencia_final_df = pd.concat([sequencia_base, evento_futuro_template], ignore_index=True)
                        
                        X_para_prever = sequencia_final_df.drop(columns=['total_vitima', 'data_referencia', 'municipio'])
                        X_processado = preprocessor.transform(X_para_prever)
                        X_final = np.reshape(X_processado, (1, X_processado.shape[0], X_processado.shape[1]))
                        
                        previsao_evento_normalizada = model.predict(X_final)
                        previsao_evento_real = y_scaler.inverse_transform(previsao_evento_normalizada)
                        vitimas_por_evento = np.ceil(previsao_evento_real[0][0])
                        
                        previsao_anual_total = vitimas_por_evento * media_eventos_ano
                    
                    st.success("Previsão Concluída!")
                    st.metric(
                        label=f"Estimativa de Vítimas para {ano_desejado}",
                        value=f"{int(previsao_anual_total)}",
                        delta_color="off"
                    )
                    st.caption(f"Cálculo baseado em uma previsão de {int(vitimas_por_evento)} vítimas por evento, multiplicado pela média de {media_eventos_ano:.1f} eventos/ano para o cenário escolhido.")