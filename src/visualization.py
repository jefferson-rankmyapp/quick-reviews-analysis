import plotly.express as px

def create_bar_chart(data, x_col, y_col, text_col, title, labels):
    """
    Gera um gráfico de barras utilizando Plotly.

    Args:
        data (pd.DataFrame): Dados a serem visualizados.
        x_col (str): Coluna para o eixo X.
        y_col (str): Coluna para o eixo Y.
        text_col (str): Coluna com os valores para exibir nos textos.
        title (str): Título do gráfico.
        labels (dict): Dicionário de rótulos para os eixos.
    
    Returns:
        plotly.graph_objs._figure.Figure: Objeto do gráfico.
    """
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        text=text_col,
        title=title,
        labels=labels,
        template='plotly_white'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    return fig

def create_pie_chart(data, values_col, names_col, title, labels):
    """
    Gera um gráfico de pizza utilizando Plotly.

    Args:
        data (pd.DataFrame): Dados a serem visualizados.
        values_col (str): Coluna com os valores numéricos.
        names_col (str): Coluna com os nomes dos segmentos.
        title (str): Título do gráfico.
        labels (dict): Dicionário de rótulos para os segmentos.

    Returns:
        plotly.graph_objs._figure.Figure: Objeto do gráfico.
    """
    fig = px.pie(
        data,
        values=values_col,
        names=names_col,
        title=title,
        labels=labels,
        template='plotly_white'
    )
    fig.update_traces(textinfo='percent+label')
    return fig
