import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from tensorflow.keras.models import load_model
import warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
import os
import json
import requests
from streamlit_plotly_events import plotly_events

#   Controle para navegação
if 'pagina_selecionada' not in st.session_state:
    st.session_state.pagina_selecionada = "📊 Dashboard de Análise"


#   Modelo de linguagem para stopwords
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    print("Baixando modelo de linguagem 'pt_core_news_sm'. Isso pode demorar um pouco...")
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load('pt_core_news_sm')

#   Página de avisos 
st.set_page_config(layout="wide", page_title="Análise de Violência no Brasil")

warnings.filterwarnings("ignore", category=FutureWarning)

#   Carregamento de arquivos de previsão
#   @st.cache_resource garante que o modelo pesado e os arquivos sejam carregados apenas uma vez.
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

#   Carrega o dataset para o dashboard e para a lógica de previsão
try:
    df_completo = pd.read_csv("Dados_2015_2024.csv")
    df_completo['data_referencia'] = pd.to_datetime(df_completo['data_referencia'], errors='coerce')
except FileNotFoundError:
    st.error("Erro: O arquivo 'Dados_2015_2024.csv' não foi encontrado. Por favor, coloque-o na mesma pasta.")
    st.stop() 


with st.sidebar:
    st.markdown("<h2 style='text-align: center; font-size: 25px; color: white'> Dados da Violência no Brasil</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        /* Espaçamento para o menu radio */
        div[role="radiogroup"] > div {
            margin-bottom: 15px;
        }

        /* --- CSS PARA OS BOTÕES DE EMOJI --- */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
            background-color: transparent; 
            border: none;                
            padding: 0 !important;        
            font-size: 24px;              
            color: white !important;      
            text-decoration: none;        
            transition: transform 0.1s ease-in-out; 
        }

        /* Efeito ao passar o mouse */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button:hover {
            transform: scale(1.2); /* Aumenta um pouco o tamanho */
            color: white;
        }
        
        /* Remove o contorno azul ao clicar */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button:focus {
            box-shadow: none !important;
            outline: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("🏠", use_container_width=True):
            st.session_state.pagina_selecionada = "📊 Dashboard de Análise"
    with col2:
        if st.button("📊", use_container_width=True):
            st.session_state.pagina_selecionada = "📊 Dashboard de Análise"
    with col3:
        if st.button("🧠", use_container_width=True):
            st.session_state.pagina_selecionada = "🧠 Módulo de Previsão"
    with col4:
        if st.button("📜", use_container_width=True):
            st.session_state.pagina_selecionada = "📜 Análise de Palavras"
    with col5:
        if st.button("⚙️", use_container_width=True):
            st.session_state.pagina_selecionada = "⚙️ Detalhes Técnicos"
    with col6:
        if st.button("ℹ️", use_container_width=True):
            st.session_state.pagina_selecionada = "ℹ️ Sobre o Projeto"
    
    st.markdown("---") 

    opcoes_menu = (
        "📊 Dashboard de Análise", "🧠 Módulo de Previsão", "📜 Análise de Palavras", 
        "⚙️ Detalhes Técnicos", "ℹ️ Sobre o Projeto"
    )
    
    st.radio(
        "Escolha uma seção:",
        opcoes_menu,
        key="pagina_selecionada"
    )

    st.markdown("---")
    
    st.info("Este painel oferece uma análise visual dos dados de violência e um módulo para estimativas futuras. ")
    
    st.markdown("""
    <div style="font-size: 12px; color: #D3D3D3; line-height: 1.6;">
        <br>
        <br>
        Trabalho de Conclusão de Curso (TCC) 
        <br>
        Gestão da Tecnologia da Informação (GTI)
        <br>
        IF Sudeste MG - Campus Muriaé
        <br>
        Desenvolvido por Flavia Santos
    </div>
    """, unsafe_allow_html=True)
    
    
#   Dashboard de Análise
if st.session_state.pagina_selecionada == "📊 Dashboard de Análise":

    df = df_completo.copy()
    df['Ano'] = df['data_referencia'].dt.year
    df['Mes'] = df['data_referencia'].dt.month_name()

    meses_pt = {
        'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
        'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
        'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
    }
    df['Mes'] = df['Mes'].map(meses_pt)

    st.markdown("<h1 style='text-align: center; font-size: 40px; color: white'>📊 Dados da Violência no Brasil</h1>", unsafe_allow_html=True)
    
    st.info("Exploração detalhada dos dados sobre a violência no Brasil. Utilize os filtros de Ano, Estado e Tipo de Evento para visualizar os gráficos e a tabela com informações específicas. Dica: ao selecionar um único estado, o filtro por cidade será habilitado para uma análise ainda mais granular.")

    # Filtros para seleção de previsão
    anos = sorted(df['Ano'].unique())
    todos_estados = sorted(df['uf'].unique())
    eventos = sorted(df['evento'].unique())

    col1, col2, col3 = st.columns(3)

    with col1:
        ano_selecionado = st.selectbox("Selecione o Ano", anos, key="ano")

    with col2:
        estado_selecionado = st.multiselect(
            "Selecione os Estados",
            options=todos_estados,
            key="estado",
            placeholder="Todos"
        )

    with col3:
        evento_input = st.selectbox("Tipo de Evento", ["Todos"] + eventos, key="evento")

    if not estado_selecionado:
        estados_filtrados = todos_estados
    else:
        estados_filtrados = estado_selecionado

    if len(estados_filtrados) == 1:
        cidades = df[df['uf'] == estados_filtrados[0]]['municipio'].sort_values().unique()
        cidade_input = st.selectbox("Selecione a Cidade", ["Todas"] + list(cidades), index=0, key="cidade")
    else:
        st.selectbox("Selecione a Cidade", ["Todas"], index=0, disabled=True, key="cidade_disabled")
        cidade_input = "Todas"

    df_filtrado = df[df['Ano'] == ano_selecionado]
    df_filtrado = df_filtrado[df_filtrado['uf'].isin(estados_filtrados)]

    if cidade_input != "Todas":
        df_filtrado = df_filtrado[df_filtrado['municipio'] == cidade_input]

    if evento_input != "Todos":
        df_filtrado = df_filtrado[df_filtrado['evento'] == evento_input]

    #   Total de vítimas pelo dataframe já filtrado
    total_vitimas = df_filtrado['total_vitima'].sum()
    #   Formata o número para ter separador de milhar (ex: 12.345)
    total_formatado = f"{total_vitimas:,}".replace(',', '.')

    if evento_input == "Todos":
        titulo_base = f"Casos de violência no Brasil - {ano_selecionado}"
    else:
        titulo_base = f"{evento_input} - {ano_selecionado}"

    titulo_final = f"{titulo_base} (Total de Vítimas: {total_formatado})"

    st.markdown(f"<h2 style='font-size: 32px; color: white; font-weight: bold !important;'>{titulo_final}</h2>", unsafe_allow_html=True)

    #   Gráfico de Barras
    st.markdown("<h3 style='font-size: 22px; color: white;'>Total de Vítimas por Estado</h3>", unsafe_allow_html=True)
    df_barra = df_filtrado.groupby('uf')['total_vitima'].sum().reset_index()

    if len(estados_filtrados) == 1:
        total_estado = df_barra['total_vitima'].iloc[0]
        df_barra['uf'] = df_barra['uf'] + f' (Total: {total_estado})'

    fig_barra = px.bar(
        df_barra,
        x='uf',
        y='total_vitima',
        text='total_vitima',
        labels={'uf': 'Estado', 'total_vitima': 'Total de Vítimas'},
        color='uf'
    )
    fig_barra.update_traces(width=0.6)
    st.plotly_chart(fig_barra)

    #   Gráfico de Linhas
    st.subheader("Evolução Mensal dos Casos por Estado")

    df_linha = df_filtrado.groupby(['uf', 'Mes'])['total_vitima'].sum().reset_index()
    ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    df_linha['Mes'] = pd.Categorical(df_linha['Mes'], categories=ordem_meses, ordered=True)
    df_linha = df_linha.sort_values(['uf', 'Mes'])

    fig_linha = px.line(
        df_linha,
        x='Mes',
        y='total_vitima',
        color='uf',
        markers=True,
        labels={
            'Mes': 'Mês',
            'total_vitima': 'Total de Vítimas',
            'uf': 'Estado'
        }
    )

    fig_linha.update_traces(textposition='top center')
    st.plotly_chart(fig_linha)
    

    # 🔹 Subtítulo
    st.subheader("Mapa Geográfico - Distribuição por Estado")

    # 🔹 GeoJSON do Brasil
    url_geojson = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    geojson_estados = requests.get(url_geojson).json()

    # 🔹 Dados
    df_mapa = df_filtrado.groupby('uf')['total_vitima'].sum().reset_index()

    # 🔹 Figura inicial
    fig_mapa = px.choropleth(
        df_mapa,
        geojson=geojson_estados,
        locations='uf',
        featureidkey="properties.sigla",
        color='total_vitima',
        color_continuous_scale="YlOrRd",
        hover_data={'uf': True, 'total_vitima': True},
        labels={'total_vitima': 'Total de Vítimas'},
        title=f"Distribuição Geográfica de Vítimas - {ano_selecionado}"
    )

    fig_mapa.update_geos(fitbounds="locations", visible=False)

    fig_mapa.update_layout(
        autosize=True,
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        coloraxis_colorbar=dict(
            title="Nº de Vítimas",
            thickness=15,
            len=0.8
        )
    )

    fig_mapa.update_traces(
        hovertemplate="<b>%{location}</b><br>Total de Vítimas: %{z}<extra></extra>",
        marker_line_width=1,
        marker_line_color="black"
    )

    # 🔹 Captura de clique no mapa
    selected_points = plotly_events(
        fig_mapa,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=600,
        override_width="100%"
    )

    # 🔹 Se clicou em algum estado, aplicar zoom só nele
    if selected_points:
        estado_clicado = selected_points[0]['location']
        st.write(f"🔎 Estado selecionado: **{estado_clicado}**")

        # Novo gráfico apenas para o estado clicado
        fig_zoom = px.choropleth(
            df_mapa[df_mapa['uf'] == estado_clicado],
            geojson=geojson_estados,
            locations='uf',
            featureidkey="properties.sigla",
            color='total_vitima',
            color_continuous_scale="YlOrRd"
        )
        fig_zoom.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_zoom, use_container_width=True)



    #   Gráfico de Pizza
    st.subheader("Distribuição de Tipos de Armas por Faixa Etária")
    col4, col5 = st.columns(2)

    with col4:
        faixa_etaria_input = st.selectbox(
            "Selecione a Faixa Etária",
            options=["Todas"] + sorted(df_filtrado['faixa_etaria'].dropna().unique().tolist()),
            key="faixa"
        )

    with col5:
        tipo_arma_input = st.selectbox(
            "Selecione o Tipo de Arma",
            options=["Todas"] + sorted(df_filtrado['arma'].dropna().unique().tolist()),
            key="arma"
        )

    df_pizza = df_filtrado.copy()
    if faixa_etaria_input != "Todas":
        df_pizza = df_pizza[df_pizza['faixa_etaria'] == faixa_etaria_input]
    if tipo_arma_input != "Todas":
        df_pizza = df_pizza[df_pizza['arma'] == tipo_arma_input]

    dados_pizza = df_pizza.groupby('arma').size().reset_index(name='quantidade')
    dados_pizza = dados_pizza.rename(columns={'arma': 'Tipo de Arma'})

    if not dados_pizza.empty:
        fig_pizza = px.pie(
            dados_pizza,
            names='Tipo de Arma',
            values='quantidade',
            title="Distribuição de Armas (Filtrada)",
            hole=0.4
        )
        st.plotly_chart(fig_pizza)
    else:
        st.warning("Nenhum dado disponível para os filtros selecionados.")

    #   Tabela
    colunas_para_mostrar = df_filtrado.drop(columns=['Ano'])
    colunas_para_mostrar = colunas_para_mostrar[
        (df_filtrado['feminino'] >= 1) |
        (df_filtrado['masculino'] >= 1) |
        (df_filtrado['nao_informado'] >= 1)
    ].copy()

    colunas_para_mostrar['data_referencia'] = pd.to_datetime(colunas_para_mostrar['data_referencia']).dt.strftime('%d-%m-%Y')

    colunas_numericas = colunas_para_mostrar.select_dtypes(include='number')
    colunas_validas = colunas_numericas.columns[colunas_numericas.sum() > 0]

    colunas_para_mostrar = pd.concat([
        colunas_para_mostrar.select_dtypes(exclude='number'),
        colunas_para_mostrar[colunas_validas]
    ], axis=1)

    colunas_para_mostrar.reset_index(drop=True, inplace=True)
    st.subheader("Dados Filtrados")
    st.dataframe(colunas_para_mostrar)

    #   Rodapé
    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")

    #   Previsão do modelo 

elif st.session_state.pagina_selecionada == "🧠 Módulo de Previsão":
    
    st.markdown("<h1 style='text-align: center; color: white;'>🧠 Módulo de Previsão Anual</h1>", unsafe_allow_html=True)
    st.markdown("#### Como Funciona?")
    st.info("""
    Este módulo utiliza um modelo de Inteligência Artificial, especificamente uma **rede neural recorrente (LSTM - Long Short-Term Memory)**, para projetar estimativas futuras. O modelo foi treinado para reconhecer padrões em sequências de eventos com base nos dados históricos de 2015 a 2024. 
    Para prever um resultado, ele analisa uma janela dos eventos mais recentes que correspondem ao cenário selecionado e, a partir dos padrões aprendidos, estima o número de vítimas. Esse valor é então extrapolado para gerar a estimativa para o ano completo.
    """)

    st.markdown("#### Como Usar?")
    st.write("""
    1.  **Clique no botão '🚀 Iniciar Nova Previsão'** para abrir o painel de parâmetros.
    2.  No campo **'Digite o ANO para a previsão'**, insira o ano futuro que deseja estimar.
    3.  Utilize os **filtros opcionais** (UF, Evento, Arma, Faixa Etária) para refinar o cenário da sua previsão. Deixar em "Todos" gera uma estimativa mais geral.
    4.  Clique em **'Calcular Estimativa'** e aguarde o modelo processar os dados.
    """)

    model, preprocessor, y_scaler = carregar_ativos_previsao()
    
    if not model:
        st.error("Arquivos de modelo não encontrados! Certifique-se de que 'melhor_modelo_multivariado.keras', 'preprocessor.joblib' e 'y_scaler.joblib' estão na pasta.")
        st.stop()
        
    if st.button("🚀 Iniciar Nova Previsão", type="primary"):
        
        @st.dialog("Parâmetros da Previsão", width="large")
        def prediction_dialog():
            st.markdown("#### Preencha os campos para gerar a estimativa:")
            
            ano_desejado = st.number_input("Digite o ANO para a previsão (Obrigatório)", min_value=df_completo['Ano'].max() + 1, value=df_completo['Ano'].max() + 1, step=1)
            
            col_filtros1, col_filtros2 = st.columns(2)
            with col_filtros1:
                uf_selecionada = st.selectbox("Filtrar por UF (Opcional)", ["Todos"] + sorted(df_completo['uf'].unique()))
                
                if uf_selecionada != "Todos":
                    cidades_disponiveis = sorted(df_completo[df_completo['uf'] == uf_selecionada]['municipio'].unique())
                    cidade_selecionada = st.selectbox("Filtrar por Cidade (Opcional)", ["Todas"] + cidades_disponiveis)
                else:
                    cidade_selecionada = "Todas"
                    
                arma_selecionada = st.selectbox("Filtrar por Arma (Opcional)", ["Todos"] + sorted(df_completo['arma'].unique()))

                

            with col_filtros2:
                evento_selecionado = st.selectbox("Filtrar por Evento (Opcional)", ["Todos"] + sorted(df_completo['evento'].unique()))
                faixa_selecionada = st.selectbox("Filtrar por Faixa Etária (Opcional)", ["Todos"] + sorted(df_completo['faixa_etaria'].unique()))

            if st.button("Calcular Estimativa"):
                df_filtrado_pred = df_completo.copy()

                if uf_selecionada != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['uf'] == uf_selecionada]
                if cidade_selecionada != "Todas": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['municipio'] == cidade_selecionada]
                if evento_selecionado != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['evento'] == evento_selecionado]
                if arma_selecionada != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['arma'] == arma_selecionada]
                if faixa_selecionada != "Todos": df_filtrado_pred = df_filtrado_pred[df_filtrado_pred['faixa_etaria'] == faixa_selecionada]
                
                # 🔹 É AQUI QUE VOCÊ ENTRA COM O TRATAMENTO:
                if df_filtrado_pred.empty:
                    st.error("❌ Nenhum dado encontrado para os filtros escolhidos. Tente opções menos específicos.")
                    return

                if len(df_filtrado_pred) < 10:
                    st.warning(f"⚠️ Dados históricos insuficientes, é necessário pelo menos {len(df_filtrado_pred)} eventos para gerar uma previsão confiável.")
                    return

                janela = 10
                if len(df_filtrado_pred) < janela:
                    st.error(f"Dados históricos insuficientes é necessário pelo menos {len(df_filtrado_pred)} eventos para o cenário. Tente filtros menos específicos.")
                else:
                    with st.spinner("Calculando... O modelo está processando os dados."):
                        num_anos_historico = df_filtrado_pred['Ano'].nunique()
                        media_eventos_ano = len(df_filtrado_pred) / num_anos_historico if num_anos_historico > 0 else 0
                        
                        sequencia_base = df_filtrado_pred.tail(janela - 1).copy()
                        evento_futuro_template = df_filtrado_pred.tail(1).copy()
                        evento_futuro_template['Ano'] = ano_desejado
                        
                        sequencia_final_df = pd.concat([sequencia_base, evento_futuro_template], ignore_index=True)

                        X_para_prever = sequencia_final_df.drop(columns=['total_vitima', 'data_referencia', 'municipio'])

                        for col in X_para_prever.select_dtypes(include=['object']).columns:
                            if col in preprocessor.feature_names_in_:
                                X_para_prever[col] = X_para_prever[col].astype('category')
                        
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
        
        prediction_dialog()

        st.markdown("---")
        st.markdown("Desenvolvido por Flavia 💙")

#   Análise de Palavras
elif st.session_state.pagina_selecionada == "📜 Análise de Palavras":

    st.markdown("<h1 style='text-align: center; color: white;'>📜 Análise de Palavras - Tipos de Evento</h1>", unsafe_allow_html=True)
    st.info("Frequência dos eventos exibidas em Nuvem de Palavras e através de uma tabela de percentual de cada evento.")

    try:
        df_frequencia_frase = pd.read_csv("Frequencia_Frases_Evento.csv")

        st.subheader("Frequência de Tipos de Evento")
        
        fator_de_escala = 0.5
        
        dicionario_frases_escalonado = dict(zip(
            df_frequencia_frase['Frase'], 
            df_frequencia_frase['Contagem'] ** fator_de_escala
        ))
        
        if not dicionario_frases_escalonado:
            st.warning("Não há dados de frequência para gerar a nuvem de palavras de eventos.")
        else:
            wordcloud_frases = WordCloud(
                width=800, height=400, background_color="black", 
                colormap="hot", collocations=False
            ).generate_from_frequencies(dicionario_frases_escalonado)

            fig_frases, ax_frases = plt.subplots(figsize=(7, 5))
            plt.style.use("dark_background")
            ax_frases.imshow(wordcloud_frases, interpolation="bilinear")
            ax_frases.axis("off")

            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                st.pyplot(fig_frases)

        with st.expander("Ver Tabela de Frequência Completa de Eventos"):
            df_frequencia_frase['Porcentagem'] = df_frequencia_frase['Porcentagem'].map('{:.2f}%'.format)
            st.dataframe(df_frequencia_frase, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("Arquivo 'Frequencia_Frases_Evento.csv' não encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro na análise de eventos: {e}")

    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")
    
    #   Sobre as Tecnologias 
    
elif st.session_state.pagina_selecionada == "⚙️ Detalhes Técnicos":

    st.markdown("<h1 style='text-align: center; color: white;'>⚙️ Detalhes Técnicos do Projeto</h1>", unsafe_allow_html=True)
    st.info("Arquitetura, tecnologias e a metodologia utilizadas para o desenvolvimento desta ferramenta de análise e previsão.")

    st.markdown("---")

    st.subheader("Tecnologias e Linguagens Utilizadas")
    st.markdown("""
    A ferramenta foi desenvolvida inteiramente na linguagem **Python**, utilizando um conjunto de bibliotecas especializadas para cada etapa do projeto:
    - **Interface Web e Dashboard:** **Streamlit**
    - **Manipulação e Análise de Dados:** **Pandas** e **NumPy**
    - **Visualização de Dados:** **Plotly Express**, **Matplotlib** e **WordCloud**
    - **Machine Learning e Modelagem Preditiva:** **TensorFlow (Keras)**, **Scikit-learn** e **Joblib**
    """)

    st.markdown("---")

    st.subheader("Modelo de Previsão: Rede Neural LSTM")
    st.write("""
    O módulo de previsão utiliza um modelo de **Rede Neural Recorrente (RNN)** do tipo **LSTM (Long Short-Term Memory)**. Este tipo de arquitetura é especialmente eficaz para analisar sequências, pois consegue "lembrar" de informações de passos anteriores para prever valores futuros.
    """)

    with st.expander("Sobre o modelo LSTM"):
        st.markdown("""
        A **Rede Neural LSTM (Long Short-Term Memory)** é um tipo avançado de Rede Neural Recorrente (RNN), projetada especificamente para aprender com sequências de dados, como séries temporais ou texto.

        Sua principal inovação em relação às RNNs tradicionais é a sua capacidade de superar o problema da "memória curta". Ela faz isso através de uma estrutura interna chamada **célula de memória**, que funciona como uma memória de longo prazo. O fluxo de informações nesta célula é controlado por três mecanismos chamados **"gates" (portões)**:

        - **Portão de Esquecimento (Forget Gate):** Analisa a nova entrada de dados e decide quais informações da memória de longo prazo devem ser descartadas.
        - **Portão de Entrada (Input Gate):** Determina quais novas informações são relevantes e devem ser armazenadas na célula de memória.
        - **Portão de Saída (Output Gate):** Filtra a memória de longo prazo para gerar a saída ou a previsão para o passo atual da sequência.
        
        Essa arquitetura permite que o modelo retenha informações por longos períodos, tornando-o ideal para capturar tendências e padrões complexos nos dados históricos de criminalidade para realizar previsões futuras.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Pré-Processamento dos Dados")
        st.markdown("""
        1.  **Codificação Categórica:** Variáveis textuais são transformadas em representações numéricas (*One-Hot Encoding*).
        2.  **Normalização:** Features numéricas são escalonadas para um intervalo entre 0 e 1.
        3.  **Criação de Janelas:** Os dados são organizados em janelas sequenciais, onde o modelo aprende a prever um resultado com base nos 10 eventos anteriores.
        """)

    with col2:
        st.markdown("##### Arquitetura e Treinamento")
        st.markdown("""
        - **Estrutura:** A rede é composta por camadas LSTM e `Dense` (Keras).
        - **Otimizador:** Foi utilizado o otimizador `Adam`.
        - **Função de Perda:** O modelo foi treinado para minimizar o Erro Quadrático Médio.
        - **Validação:** Os dados foram divididos em conjuntos de treino e teste para garantir a generalização do modelo para dados não vistos.
        """)
        
    st.markdown("---") 
    
    st.subheader("Versionamento e Deploy Contínuo")
    st.markdown("""
    Todo o código-fonte e os ativos do projeto são gerenciados e versionados com **Git** e estão hospedados em um repositório no **GitHub** chamado `dados-violencia-brasil`.

    **Principais Arquivos no Repositório:**
    - `app.py`: O script principal da aplicação Streamlit.
    - `requirements.txt`: Lista de todas as bibliotecas Python necessárias para o projeto.
    - `Dados_2015_2024.csv`: O arquivo de dados utilizado nos dashboards.
    - `Frequencia_Frases_Evento.csv`: Arquivo pré-processado com as contagens de eventos para a análise de palavras.
    - `melhor_modelo_multivariado.keras`: O arquivo do modelo de rede neural treinado.
    - `.joblib`: Arquivos dos pré-processadores de dados.
    """)
    
    st.markdown("""
    A aplicação web é publicada e hospedada na plataforma **Streamlit Community Cloud**. Este serviço está diretamente conectado ao repositório do GitHub, o que permite um fluxo de **Integração Contínua e Deploy Contínuo (CI/CD)**.
    
    Qualquer atualização enviada (`git push`) do ambiente de desenvolvimento (VS Code) para o GitHub é automaticamente detectada pelo Streamlit Cloud, que atualiza a aplicação na web de forma síncrona. Isso garante que a versão online esteja sempre refletindo o código mais recente.
    
    **A aplicação pode ser acessada publicamente no endereço:**
    [https://dados-violencia-brasil-2015-a-2024.streamlit.app/](https://dados-violencia-brasil-2015-a-2024.streamlit.app/)
    """)

    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")
    

#   Sobre o Projeto
elif st.session_state.pagina_selecionada == "ℹ️ Sobre o Projeto":

    st.markdown("<h1 style='text-align: center; color: white;'>ℹ️ Sobre o Projeto e a Fonte dos Dados</h1>", unsafe_allow_html=True)
    st.info("Visualização e análises dos dados abertos sobre segurança pública no Brasil, com o objetivo de promover a transparência e facilitar o entendimento sobre o tema.")

    st.markdown("---")

    st.subheader("Fonte dos Dados")
    st.markdown("""
    Os dados utilizados neste projeto foram coletados do portal do **Ministério da Justiça e Segurança Pública (MJSP)**, através da Secretaria Nacional de Segurança Pública (SENASP). 
    
    As informações são provenientes do **Sistema Nacional de Informações de Segurança Pública (SINESP)** e cobrem o período de **2015 a 2024**.
    
    **Links Oficiais:**
    - **Portal Principal:** [Ministério da Justiça e Segurança Pública](https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica)
    - **Base de Dados Específica:** [Dados Nacionais - SINESP VDE](https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/estatistica/dados-nacionais-1/base-de-dados-e-notas-metodologicas-dos-gestores-estaduais-sinesp-vde-2022-e-2023)
    """)

    st.markdown("---")

    st.subheader("O Mapa da Segurança Pública")
    st.write("""
    O Mapa da Segurança Pública, uma publicação anual do MJSP, representa um avanço significativo na gestão e transparência dos dados de segurança pública no Brasil. Ele sistematiza e publiciza, de forma organizada, os principais indicadores criminais e estatísticas coletadas em âmbito nacional, servindo como referência para a formulação de políticas públicas, diagnósticos e pesquisas. 
    
    Além disso, reforça o compromisso do Ministério com a divulgação regular e padronizada dessas informações, promovendo maior transparência e apoio à tomada de decisões estratégicas na área da segurança.
    """)
    
    st.markdown("---")

    with st.expander("Clique para ler sobre as atribuições da Secretaria Nacional de Segurança Pública (SENASP)"):
        st.write("""
        A Secretaria Nacional de Segurança Pública – SENASP foi criada pelo Decreto nº 2.315, de 4 de setembro de 1997.
        
        A SENASP é responsável por formular políticas, diretrizes e ações para a segurança pública no país. Possui como objetivo promover a integração e a coordenação entre as diferentes esferas governamentais e agências de segurança para enfrentar desafios relacionados à segurança pública, como a prevenção de crimes, combate à violência e capacitação de profissionais da área.
        
        Compete à SENASP o assessoramento técnico ao Ministro da Justiça, integrando os entes federativos e os órgãos que compõem o Sistema Único de Segurança Pública (SUSP), além de promover a gestão do Fundo Nacional de Segurança Pública (FNSP).
        """)

    st.markdown("---")
    st.markdown("Desenvolvido por Flavia 💙")