import streamlit as st
import pandas as pd
import plotly.express as px

# Carrega os dados
df = pd.read_csv("Banco_Dados_2015_2024.csv")

# Adiciona colunas de Ano e Mês
df['data_referencia'] = pd.to_datetime(df['data_referencia'])
df['Ano'] = df['data_referencia'].dt.year
df['Mes'] = df['data_referencia'].dt.month_name()

# Traduz nomes dos meses
meses_pt = {
    'January': 'Janeiro', 'February': 'Fevereiro', 'March': 'Março', 'April': 'Abril',
    'May': 'Maio', 'June': 'Junho', 'July': 'Julho', 'August': 'Agosto',
    'September': 'Setembro', 'October': 'Outubro', 'November': 'Novembro', 'December': 'Dezembro'
}
df['Mes'] = df['Mes'].map(meses_pt)

# ---------- TÍTULO GLOBAL NO TOPO ----------
st.markdown("<h1 style='text-align: center; font-size: 40px; color: white'>📊 Dados da Violência no Brasil</h1>", unsafe_allow_html=True)

# Filtros disponíveis
anos = sorted(df['Ano'].unique())
todos_estados = sorted(df['uf'].unique())
eventos = sorted(df['evento'].unique())

# Filtros
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

# Estados selecionados
if not estado_selecionado:
    estados_filtrados = todos_estados
else:
    estados_filtrados = estado_selecionado

# Filtro de cidade
if len(estados_filtrados) == 1:
    cidades = df[df['uf'] == estados_filtrados[0]]['municipio'].sort_values().unique()
    cidade_input = st.selectbox("Selecione a Cidade", ["Todas"] + list(cidades), index=0, key="cidade")
else:
    st.selectbox("Selecione a Cidade", ["Todas"], index=0, disabled=True, key="cidade_disabled")
    cidade_input = "Todas"

# Filtragem principal
df_filtrado = df[df['Ano'] == ano_selecionado]
df_filtrado = df_filtrado[df_filtrado['uf'].isin(estados_filtrados)]

if cidade_input != "Todas":
    df_filtrado = df_filtrado[df_filtrado['municipio'] == cidade_input]

if evento_input != "Todos":
    df_filtrado = df_filtrado[df_filtrado['evento'] == evento_input]

# ---------- TÍTULO ESPECÍFICO DOS FILTROS ----------
if evento_input == "Todos":
    titulo = f"Casos de violência no Brasil - {ano_selecionado}"
else:
    titulo = f"{evento_input} - {ano_selecionado}"
st.markdown(f"<h2 style='font-size: 36px; color: white; font-weight: bold !important;'>{titulo}</h2>", unsafe_allow_html=True)

# ---------- GRÁFICO DE BARRAS ----------
st.markdown("<h3 style='font-size: 22px; color: white;'>Total de Vítimas por Estado</h3>", unsafe_allow_html=True)
df_barra = df_filtrado.groupby('uf')['total_vitima'].sum().reset_index()

# Se só 1 estado foi selecionado, mostra o total geral no label
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
fig_barra.update_traces(textposition='outside')
st.plotly_chart(fig_barra)

# ---------- GRÁFICO DE LINHA (por Estado) ----------
st.subheader("Evolução Mensal dos Casos por Estado")

# Agrupa por estado e mês
df_linha = df_filtrado.groupby(['uf', 'Mes'])['total_vitima'].sum().reset_index()

# Garante a ordem correta dos meses
ordem_meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
               'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
df_linha['Mes'] = pd.Categorical(df_linha['Mes'], categories=ordem_meses, ordered=True)
df_linha = df_linha.sort_values(['uf', 'Mes'])

# Cria gráfico com uma linha por estado
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

# ---------- GRÁFICO DE PIZZA COM FILTROS ----------

st.subheader("Distribuição de Tipos de Armas por Faixa Etária")

# Filtros adicionais
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

# Aplica filtros de faixa etária e arma
df_pizza = df_filtrado.copy()

if faixa_etaria_input != "Todas":
    df_pizza = df_pizza[df_pizza['faixa_etaria'] == faixa_etaria_input]

if tipo_arma_input != "Todas":
    df_pizza = df_pizza[df_pizza['arma'] == tipo_arma_input]

# Agrupa dados para gráfico
dados_pizza = df_pizza.groupby('arma').size().reset_index(name='quantidade')
dados_pizza = dados_pizza.rename(columns={'arma': 'Tipo de Arma'})

# Só gera gráfico se tiver dados
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


# ---------- TABELA ----------

# Filtra as colunas que você quer mostrar, removendo as indesejadas
colunas_para_mostrar = df_filtrado.drop(columns=['Ano']);

# Filtrar apenas os dados com pelo menos uma vítima
colunas_para_mostrar[
    (df_filtrado['feminino'] >= 1) | 
    (df_filtrado['masculino'] >= 1) | 
    (df_filtrado['nao_informado'] >= 1)
].copy()

# Formatar a data antes de qualquer soma
colunas_para_mostrar['data_referencia'] = pd.to_datetime(colunas_para_mostrar['data_referencia']).dt.strftime('%d-%m-%Y')

# Selecionar apenas as colunas numéricas para a filtragem por soma > 0
colunas_numericas = colunas_para_mostrar.select_dtypes(include='number')
colunas_validas = colunas_numericas.columns[colunas_numericas.sum() > 0]

# Adiciona de volta as colunas não numéricas para exibir
colunas_para_mostrar = pd.concat([
    colunas_para_mostrar.select_dtypes(exclude='number'),
    colunas_para_mostrar[colunas_validas]
], axis=1)

# Resetar índice
colunas_para_mostrar.reset_index(drop=True, inplace=True)

# Exibir no Streamlit
st.subheader("Dados Filtrados")
st.dataframe(colunas_para_mostrar)



# ---------- RODAPÉ ----------
st.markdown("---")
st.markdown("Desenvolvido por Flavia 💙")
