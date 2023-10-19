import streamlit as st
import pandas as pd
import plotly.express as px
import locale
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz 
from sklearn.tree import export_graphviz


# Set the locale to display the currency symbol correctly
locale.setlocale(locale.LC_ALL, 'en_US')

def get_bar_chart_data(df):
    # Group by both 'state' and 'profitable_project'
    grouped_df = df.groupby(['state', 'profitable_project'])
    # Get counts for each combination of 'state' and 'profitable_project'
    counts = grouped_df.size().unstack(fill_value=0)
    # Calculate the percentages for profitable and non-profitable projects for each state
    percentages = counts.divide(counts.sum(axis=1), axis=0) * 100
    return counts, percentages


def get_bar_chart_data_by_project_group(df):
    # Group by both 'proj_group' and 'profitable_project'
    grouped_df = df.groupby(['proj_group', 'profitable_project'])
    # Get counts for each combination of 'proj_group' and 'profitable_project'
    counts = grouped_df.size().unstack(fill_value=0)
    # Calculate the percentages for profitable and non-profitable projects for each project group
    percentages = counts.divide(counts.sum(axis=1), axis=0) * 100
    return counts, percentages

def get_bar_chart_data_by_client_name(df):
    # Group by both 'client_name' and 'profitable_project'
    grouped_df = df.groupby(['client_name', 'profitable_project'])
    # Get counts for each combination of 'client_name' and 'profitable_project'
    counts = grouped_df.size().unstack(fill_value=0)
    # Calculate the percentages for profitable and non-profitable projects for each client
    percentages = counts.divide(counts.sum(axis=1), axis=0) * 100
    return counts, percentages


def get_bar_chart_data_by_category_id(df):
    # Group by both 'CATEGORYID' and 'profitable_project'
    grouped_df = df.groupby(['CATEGORYID', 'profitable_project'])
    # Get counts for each combination of 'CATEGORYID' and 'profitable_project'
    counts = grouped_df.size().unstack(fill_value=0)
    # Calculate the percentages for profitable and non-profitable projects for each CATEGORYID
    percentages = counts.divide(counts.sum(axis=1), axis=0) * 100
    return counts, percentages



# Read the projects and cities dataframes
projects = pd.read_excel('projects.xlsx')
projects['profitable_project'] = projects['profitable_project'].replace({1: 'yes', 0: 'no'})
cities = pd.read_csv('cities_completed.csv', decimal= ',')
cities = cities[['city_code', 'latitude', 'longitude', 'population', 'area', 'state']]
# Merge the two dataframes on the city_code column
merged = pd.merge(projects, cities, on='city_code', how='left')
merged['project_cash_flow'] = merged['total_income']-merged['total_outcome']


st.set_page_config(page_title="Diferenciación de torres por rentabilidad", page_icon="📡", layout="centered", initial_sidebar_state="expanded")


# Create a lateral bar
with st.sidebar:
    # Create a menu
    st.image("cell-tower.png", width=100, use_column_width=False)
    menu = st.sidebar.radio("Menu", ["Home", "Proyectos NO rentables", "Proyectos rentables", "Oportunidades"])
    st.write("---") 
# Display the selected page
if menu == "Home":
    # Set the title of the app
    st.header("Segmentación de torres por rentabilidad: La clave para optimizar estrategia y maximizar beneficios")

    # State filter on sidebar
    selected_states = st.sidebar.multiselect('Select States:', merged['state'].unique(), default=merged['state'].unique())
    selected_proj_group = st.sidebar.multiselect('Select Project Group:', merged['proj_group'].unique(), default=merged['proj_group'].unique())
    selected_company = st.sidebar.multiselect('Select Company:', merged['company'].unique(), default=merged['company'].unique())
    selected_client_name = st.sidebar.multiselect('Select Client Name:', merged['client_name'].unique(), default=merged['client_name'].unique())
    selected_project_category = st.sidebar.multiselect('Select Project Category:', merged['CATEGORYID'].unique(), default=merged['CATEGORYID'].unique())


    # Filter the merged dataframe based on selected values
    filtered_merged = merged[
        (merged['state'].isin(selected_states)) &
        (merged['proj_group'].isin(selected_proj_group)) &
        (merged['company'].isin(selected_company)) &
        (merged['client_name'].isin(selected_client_name)) &
        (merged['CATEGORYID'].isin(selected_project_category))
    ]

    # Create a map showing the profitable and non-profitable projects
    fig = px.scatter_mapbox(
        filtered_merged,
        lat='latitude',
        lon='longitude',
        color='profitable_project',
        hover_name='project_name',
        zoom=5,
        title = 'Proyectos Rentables y NO Rentables por Ubicación',
        color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'}
    )
    fig.update_layout(mapbox_style="carto-positron")
    # Display the map
    st.plotly_chart(fig)

    
    st.write("---") 




    counts, percentages = get_bar_chart_data(filtered_merged)
    fig_bar = px.bar(counts.reset_index(), x='state', y=['yes', 'no'],
                        barmode='group',
                        labels={'value': 'Number of Projects', 'variable': 'Profitable Project', 'state': 'State'},
                        title="# Proyectos Rentables y NO Rentables por Departamento",
                        color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'})
    st.plotly_chart(fig_bar)



    counts, percentages = get_bar_chart_data_by_project_group(filtered_merged)
    fig_bar = px.bar(
        counts.reset_index(), 
        x='proj_group', 
        y=['yes', 'no'],
        barmode='group',
        title="# Proyectos Rentables y NO Rentables por Grupo de Proyecto",
        color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'})
    st.plotly_chart(fig_bar)


    counts, percentages = get_bar_chart_data_by_client_name(filtered_merged)
    fig_bar = px.bar(
        counts.reset_index(), 
        x='client_name', 
        y=['yes', 'no'],
        barmode='group',
        title="# Proyectos Rentables y NO Rentables por Cliente",
        color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'})
    st.plotly_chart(fig_bar)


    counts, percentages = get_bar_chart_data_by_category_id(filtered_merged)
    fig_bar = px.bar(counts.reset_index(), x='CATEGORYID', y=['yes', 'no'],
                            barmode='group',
                            labels={'value': 'Number of Projects', 'variable': 'Profitable Project', 'CATEGORYID': 'CATEGORYID'},
                            title="# Proyectos Rentables y NO Rentables por Categoría",
                            color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'})
    st.plotly_chart(fig_bar)





elif menu == "Proyectos NO rentables":
    st.header("Identificar y Analizar Fuentes Generadoras de Costos en la Empresa")
    # State filter on sidebar
    selected_states = st.sidebar.multiselect('Select States:', merged['state'].unique(), default=merged['state'].unique())
    selected_proj_group = st.sidebar.multiselect('Select Project Group:', merged['proj_group'].unique(), default=merged['proj_group'].unique())
    selected_company = st.sidebar.multiselect('Select Company:', merged['company'].unique(), default=merged['company'].unique())
    selected_client_name = st.sidebar.multiselect('Select Client Name:', merged['client_name'].unique(), default=merged['client_name'].unique())
    selected_project_category = st.sidebar.multiselect('Select Project Category:', merged['CATEGORYID'].unique(), default=merged['CATEGORYID'].unique())


    # Filter the merged dataframe based on selected values
    filtered_merged = merged[
        (merged['state'].isin(selected_states)) &
        (merged['proj_group'].isin(selected_proj_group)) &
        (merged['company'].isin(selected_company)) &
        (merged['client_name'].isin(selected_client_name)) &
        (merged['CATEGORYID'].isin(selected_project_category)) &
        (merged['profitable_project'] == 'no')
    ]


    st.write("---") 
    # Count the number of projects in the filtered dataframe
    num_projects = len(filtered_merged)

    # Calculate the sum of cashflow for the filtered projects
    total_cashflow = filtered_merged['project_cash_flow'].sum()

    # Calculate the sum of outcome for the filtered projects
    total_outcome = -filtered_merged['total_outcome'].sum()

    # Display the number of projects, total cashflow, and total outcome
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'**Número de proyectos:** {num_projects}')
    with col2:
        st.markdown(f'**Total project cashflow:** {locale.currency(total_cashflow, grouping=True)}')
    with col3:
        st.markdown(f'**Total outcome:** {locale.currency(total_outcome, grouping=True)}')


    st.write("---") 

    # Create a slider to select the number of rows to display in the table
    num_rows_to_display = st.slider('Selecciona el top de proyectos con mayor costo:', min_value=1, max_value=len(filtered_merged), value=3)

    # Display the filtered dataframe, limited to the selected number of rows
    st.table(filtered_merged[['id', 'project_name', 'proj_group', 'project_creation_date', 'city_name', 'total_income', 'total_outcome', 'project_cash_flow']].sort_values(by=['project_cash_flow'], ascending=True).head(num_rows_to_display))


    st.write("---") 
    # Agrupar los datos por estado y sumar los flujos de efectivo para cada estado, usando valores absolutos
    state_cashflow_abs = filtered_merged.groupby("state")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    state_cashflow_abs = state_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(state_cashflow_abs, labels=state_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Departamento (Valores Absolutos)')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



    st.write("---") 
    # Agrupar los datos por grupo de proyecto y sumar los flujos de efectivo para cada uno, usando valores absolutos
    proj_group_cashflow_abs = filtered_merged.groupby("proj_group")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    proj_group_cashflow_abs = proj_group_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(proj_group_cashflow_abs, labels=proj_group_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Grupo de Proyecto (Valores Absolutos)')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)




    st.write("---") 
    # Agrupar los datos por cliente y sumar los flujos de efectivo para cada uno, usando valores absolutos
    client_cashflow_abs = filtered_merged.groupby("client_name")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    client_cashflow_abs = client_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(client_cashflow_abs, labels=client_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Cliente (Valores Absolutos)')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



    st.write("---") 
    category_cashflow_abs = filtered_merged.groupby("CATEGORYID")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    category_cashflow_abs = category_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(category_cashflow_abs, labels=category_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Categoría de Proyecto (Valores Absolutos)')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)




    # Create a map showing the profitable and non-profitable projects
    fig = px.scatter_mapbox(
        filtered_merged,
        lat='latitude',
        lon='longitude',
        color='profitable_project',
        hover_name='project_name',
        zoom=5,
        title = 'Proyectos NO Rentables por Ubicación',
        color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'}
    )
    fig.update_layout(mapbox_style="carto-positron")
    # Display the map
    st.plotly_chart(fig)






elif menu == "Proyectos rentables":
    st.header("Identificar y Analizar Fuentes Generadoras de Beneficio en la Empresa")
    # State filter on sidebar
    selected_states = st.sidebar.multiselect('Select States:', merged['state'].unique(), default=merged['state'].unique())
    selected_proj_group = st.sidebar.multiselect('Select Project Group:', merged['proj_group'].unique(), default=merged['proj_group'].unique())
    selected_company = st.sidebar.multiselect('Select Company:', merged['company'].unique(), default=merged['company'].unique())
    selected_client_name = st.sidebar.multiselect('Select Client Name:', merged['client_name'].unique(), default=merged['client_name'].unique())
    selected_project_category = st.sidebar.multiselect('Select Project Category:', merged['CATEGORYID'].unique(), default=merged['CATEGORYID'].unique())


    # Filter the merged dataframe based on selected values
    filtered_merged = merged[
        (merged['state'].isin(selected_states)) &
        (merged['proj_group'].isin(selected_proj_group)) &
        (merged['company'].isin(selected_company)) &
        (merged['client_name'].isin(selected_client_name)) &
        (merged['CATEGORYID'].isin(selected_project_category)) &
        (merged['profitable_project'] == 'yes')
    ]


    st.write("---") 
    # Count the number of projects in the filtered dataframe
    num_projects = len(filtered_merged)

    # Calculate the sum of cashflow for the filtered projects
    total_cashflow = filtered_merged['project_cash_flow'].sum()

    # Calculate the sum of outcome for the filtered projects
    total_income = filtered_merged['total_income'].sum()

    # Display the number of projects, total cashflow, and total outcome
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'**Número de proyectos:** {num_projects}')
    with col2:
        st.markdown(f'**Total project cashflow:** {locale.currency(total_cashflow, grouping=True)}')
    with col3:
        st.markdown(f'**Total income:** {locale.currency(total_income, grouping=True)}')


    st.write("---") 

    # Create a slider to select the number of rows to display in the table
    num_rows_to_display = st.slider('Selecciona el top de proyectos con mayor beneficio:', min_value=1, max_value=len(filtered_merged), value=3)

    # Display the filtered dataframe, limited to the selected number of rows
    st.table(filtered_merged[['id', 'project_name', 'proj_group', 'project_creation_date', 'city_name', 'total_income', 'total_outcome', 'project_cash_flow']].sort_values(by=['project_cash_flow'], ascending=False).head(num_rows_to_display))


    st.write("---") 
    # Agrupar los datos por estado y sumar los flujos de efectivo para cada estado, usando valores absolutos
    state_cashflow_abs = filtered_merged.groupby("state")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    state_cashflow_abs = state_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(state_cashflow_abs, labels=state_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Departamento')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



    st.write("---") 
    # Agrupar los datos por grupo de proyecto y sumar los flujos de efectivo para cada uno, usando valores absolutos
    proj_group_cashflow_abs = filtered_merged.groupby("proj_group")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    proj_group_cashflow_abs = proj_group_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(proj_group_cashflow_abs, labels=proj_group_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Grupo de Proyecto')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)




    st.write("---") 
    # Agrupar los datos por cliente y sumar los flujos de efectivo para cada uno, usando valores absolutos
    client_cashflow_abs = filtered_merged.groupby("client_name")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    client_cashflow_abs = client_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(client_cashflow_abs, labels=client_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Cliente')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)



    st.write("---") 
    category_cashflow_abs = filtered_merged.groupby("CATEGORYID")["project_cash_flow"].sum().abs()
    # Ordenar los valores de mayor a menor
    category_cashflow_abs = category_cashflow_abs.sort_values(ascending=False)
    # Paleta de colores que coincida aproximadamente con Streamlit
    streamlit_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    # Crear un diagrama de pastel
    fig, ax = plt.subplots()
    ax.pie(category_cashflow_abs, labels=category_cashflow_abs.index, autopct='%1.1f%%', startangle=90, colors=streamlit_palette)
    # Dibujar un círculo en el centro para hacerlo un donut chart
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)
    # Asegurarse de que el gráfico se muestra como un círculo y no como una elipse
    ax.axis('equal')
    # Título del gráfico
    plt.title('Participación en Cashflow por Categoría de Proyecto')
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)


    # Create a map showing the profitable and non-profitable projects
    fig = px.scatter_mapbox(
        filtered_merged,
        lat='latitude',
        lon='longitude',
        color='profitable_project',
        hover_name='project_name',
        zoom=5,
        title = 'Proyectos NO Rentables por Ubicación',
        color_discrete_map={'yes': 'rgb(46,199,192)', 'no': 'rgb(242,75,75)'}
    )
    fig.update_layout(mapbox_style="carto-positron")
    # Display the map
    st.plotly_chart(fig)


elif menu == "Oportunidades":
    st.header("Palancas para Incentivar la Rentabilidad de los Proyectos")

    # st.table(merged.head(5))
    merged_no_nulls = merged.dropna()


    # Eliminar columnas no deseadas y asignar variables independientes y dependiente
    X = merged_no_nulls.drop(['project_cash_flow','longitude', 'latitude', 'total_outcome', 'total_income', 
                            'sum_qty', 'num_negative_transactions', 'num_bilateral_transactions', 
                            'num_neutral_transactions', 'num_outcome_transactions', 
                            'num_income_transactions', 'num_transactions', 'num_sub_projects', 
                            'country', 'client_code' , 'city_name', 'city_code', 'company', 
                            'project_name', 'profitable_project', 'CATEGORYID', 'id', 'project_creation_date'], axis=1)
    y = merged_no_nulls['profitable_project']

    # Aplicar get_dummies para convertir variables categóricas en dummy/indicator variables
    X = pd.get_dummies(X, columns=['proj_group', 'client_name', 'state'])

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # Inicializar y entrenar el clasificador de árboles de decisión
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Crear datos DOT para la visualización del árbol
    dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, 
                            feature_names=X.columns, class_names=['Not Profitable', 'Profitable'])

    # Crear una fuente Graphviz desde los datos DOT
    graph = graphviz.Source(dot_data)

    # Mostrar el árbol de decisión usando Streamlit
    st.header('Decision Tree')
    st.graphviz_chart(dot_data)
