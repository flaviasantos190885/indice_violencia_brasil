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
    df_completo['data_referencia'] = pd.to_datetime(df_completo['data_referencia'], errors='coerce')
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
# --- SEÇÃO 1: DASHBOARD DE ANÁLISE (COM AS CORREÇÕES) ---
# ==============================================================================
if pagina_selecionada == "Dashboard de Análise":
    
    st.markdown("<h1 style='text-align: center; color: white;'>📊 Dashboard da Violência no Brasil</h1>", unsafe_allow_html=True)
    st.markdown("---")

    df = df_completo.copy()
    df['Ano'] = df['data_referencia'].dt.year
    df['Mês'] = df['data_referencia'].dt.strftime('%B') # Usar strftime para pegar o nome completo do mês
    meses_pt = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    df['Mês'] = df['Mês'].map(meses_pt)
    
    # Seus filtros e gráficos do dashboard original...
    anos = sorted(df['Ano'].unique())
    todos_estados = sorted(df['uf'].unique())
    eventos = sorted(df['evento'].unique())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ano_selecionado = st.selectbox("Selecione o Ano", anos)
    with col2:
        # Mudado para um selectbox para simplicidade, já que o multiselect pode ser complexo com o gráfico de linha
        estado_selecionado = st.selectbox("Selecione o Estado", ["Todos os Estados"] + todos_estados)
    with col3:
        evento_input = st.selectbox("Tipo de Evento", ["Todos"] + eventos)
    
    # A lógica para a cidade aparecer condicionalmente foi mantida
    cidade_input = "Todas"
    if estado_selecionado != "Todos os Estados":
        cidades = sorted(df[df['uf'] == estado_selecionado]['municipio'].unique())
        cidade_input = st.selectbox("Selecione a Cidade", ["Todas"] + cidades)

    # Aplicação dos filtros
    df_filtrado = df[df['Ano'] == ano_selecionado]
    if estado_selecionado != "Todos os Estados":
        df_filtrado = df_filtrado[df_filtrado['uf'] == estado_selecionado]
    if cidade_input != "Todas":
        df_filtrado = df_filtrado[df_filtrado['municipio'] == cidade_input]
    if evento_input != "Todos":
        df_filtrado = df_filtrado[df_filtrado['evento'] == evento_input]
    
    # --- GRÁFICO DE BARRAS ---
    st.markdown("### Total de Vítimas por Estado")
    df_barra = df_filtrado.groupby('uf')['total_vitima'].sum().reset_index()
    fig_barra = px.bar(df_barra, x='uf', y='total_vitima', text_auto=True, labels={'uf': 'Estado', 'total_vitima': 'Total de Vítimas'}, color='uf')
    
    # --- CORREÇÃO 1: Adicionando esta linha para deixar as barras mais largas ---
    fig_barra.update_layout(bargap=0.1) # Experimente valores entre 0 e 0.2

    st.plotly_chart(fig_barra, use_container_width=True)

    # --- GRÁFICO DE LINHA ---
    st.markdown("### Evolução Mensal dos Casos")
    
    # --- CORREÇÃO 2: Lógica robusta de preparação de dados para o gráfico de linha ---
    # Usamos o df_filtrado da seleção principal para manter a consistência
    df_para_linha = df_filtrado.copy()
    
    # Agrupamos os dados por Mês e Estado
    df_evolucao_mensal = df_para_linha.groupby(['Mês', 'uf'])['total_vitima'].sum().reset_index()

    # Definimos a ordem correta dos meses para o eixo X
    ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho','Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    df_evolucao_mensal['Mês'] = pd.Categorical(df_evolucao_mensal['Mês'], categories=ordem_meses, ordered=True)

    # Ordenamos os dados pelo mês para que as linhas sejam desenhadas corretamente
    df_evolucao_mensal = df_evolucao_mensal.sort_values('Mês')

    # Criamos o gráfico de linha com os dados preparados
    fig_linha = px.line(
        df_evolucao_mensal,
        x='Mês',
        y='total_vitima',
        color='uf', # Mantém uma linha por estado no dataframe filtrado
        markers=True,
        labels={'total_vitima': 'Total de Vítimas', 'Mês': 'Mês', 'uf': 'Estado'},
        title='Evolução Mensal dos Casos por Estado' if estado_selecionado == "Todos os Estados" else f'Evolução Mensal dos Casos em {estado_selecionado}'
    )
    st.plotly_chart(fig_linha, use_container_width=True)

    # --- TABELA DE DADOS ---
    st.markdown("### Dados Filtrados")
    st.dataframe(df_filtrado.drop(columns=['Ano', 'Mês']))

# ==============================================================================
# --- SEÇÃO 2: MÓDULO DE PREVISÃO (SEU CÓDIGO JÁ CORRIGIDO) ---
# ==============================================================================
elif pagina_selecionada == "Módulo de Previsão":
    # Cole aqui a versão completa e corrigida da Seção 2 que te enviei anteriormente.
    # Se precisar dela novamente, é só pedir!
    st.markdown("<h1 style='text-align: center; color: white;'>🧠 Módulo de Previsão Anual</h1>", unsafe_allow_html=True)
    # ... (Restante do código da Seção 2) ...
# ==============================================================================
# --- SEÇÃO 2: MÓDULO DE PREVISÃO (VERSÃO COMPLETA E CORRIGIDA) ---
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
        
        # A sintaxe correta do st.dialog usa um decorador em uma função
        @st.dialog("Parâmetros da Previsão", width="large")
        def prediction_dialog():
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

                # Lógica de previsão
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
                        
                        # --- EXPLICAÇÃO DA MUDANÇA ---
                        # A correção do erro anterior está aqui. A ordem das linhas foi trocada.
                        
                        # 1. PRIMEIRO, criamos o DataFrame 'X_para_prever'
                        X_para_prever = sequencia_final_df.drop(columns=['total_vitima', 'data_referencia', 'municipio'])

                        # 2. DEPOIS, com a variável já criada, fazemos o loop para ajustar os tipos
                        for col in X_para_prever.select_dtypes(include=['object']).columns:
                            if col in preprocessor.feature_names_in_:
                                X_para_prever[col] = X_para_prever[col].astype('category')
                        
                        # Continuação da lógica...
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
                # st.caption(f"Cálculo baseado em uma previsão de {int(vitimas_por_evento)} vítimas por evento, multiplicado pela média de {media_eventos_ano:.1f} eventos/ano para o cenário escolhido.")
        
        # Esta linha chama a função que definimos acima, fazendo o dialog aparecer
        prediction_dialog()