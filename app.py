import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from src.visualization import create_bar_chart, create_pie_chart

# Baixar stopwords do nltk
nltk.download("stopwords")
stop_words_pt = set(stopwords.words("portuguese"))

# Função para extrair palavras-chave e bigramas
def extract_keywords_and_bigrams(text_series, top_n=10):
    if text_series.dropna().empty:
        return []  # Retorna vazio se não houver texto válido
    
    stop_words_list = list(stop_words_pt)  # Converte o conjunto de stopwords para uma lista
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stop_words_list)
    ngram_counts = vectorizer.fit_transform(text_series.dropna())
    ngram_sums = ngram_counts.sum(axis=0)
    ngram_freq = [(word, int(ngram_sums[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
    return Counter(dict(ngram_freq)).most_common(top_n)

# Função para carregar o arquivo e processar os dados
def process_data(file):
    df = pd.read_excel(file)
    # Total de Reviews e Média Geral (Rating)
    total_reviews = df.shape[0]
    average_rating = df['Rating'].mean()

    # Total e Percentual por Channel
    channel_distribution = df['Channel'].value_counts(normalize=True).reset_index()
    channel_distribution.columns = ['Channel', 'Percentage']
    channel_distribution['Count'] = df['Channel'].value_counts().values

    # Total e Percentual por Status
    status_distribution = df['Status'].value_counts(normalize=True).reset_index()
    status_distribution.columns = ['Status', 'Percentage']
    status_distribution['Count'] = df['Status'].value_counts().values

    # Distribuição por Rating
    rating_distribution = df['Rating'].value_counts(normalize=True).reset_index()
    rating_distribution.columns = ['Rating', 'Percentage']
    rating_distribution['Count'] = df['Rating'].value_counts().values

    # Distribuição por Sentiment
    sentiment_distribution = df['Sentiment'].value_counts(normalize=True).reset_index()
    sentiment_distribution.columns = ['Sentiment', 'Percentage']
    sentiment_distribution['Count'] = df['Sentiment'].value_counts().values

    # Distribuições por Category, Subcategory, Detailing
    category_distribution = df['Category'].value_counts(normalize=True).reset_index()
    category_distribution.columns = ['Category', 'Percentage']
    category_distribution['Count'] = df['Category'].value_counts().values

    subcategory_distribution = df['Subcategory'].value_counts(normalize=True).reset_index()
    subcategory_distribution.columns = ['Subcategory', 'Percentage']
    subcategory_distribution['Count'] = df['Subcategory'].value_counts().values

    detailing_distribution = df['Detailing'].value_counts(normalize=True).reset_index()
    detailing_distribution.columns = ['Detailing', 'Percentage']
    detailing_distribution['Count'] = df['Detailing'].value_counts().values

    channel_distribution['Percentage'] *= 100
    status_distribution['Percentage'] *= 100
    rating_distribution['Percentage'] *= 100
    sentiment_distribution['Percentage'] *= 100
    category_distribution['Percentage'] *= 100
    subcategory_distribution['Percentage'] *= 100
    detailing_distribution['Percentage'] *= 100

    # Análise textual
    positive_reviews = df[df['Sentiment'].str.lower().isin(['positivo', 'positive'])]['Review']
    negative_reviews = df[df['Sentiment'].str.lower().isin(['negativo', 'negative'])]['Review']
    positive_keywords_bigrams = extract_keywords_and_bigrams(positive_reviews)
    negative_keywords_bigrams = extract_keywords_and_bigrams(negative_reviews)

    return {
        "Total Reviews": total_reviews,
        "Average Rating": average_rating.round(2),
        "Channel Distribution": channel_distribution.round(2),
        "Status Distribution": status_distribution.round(2),
        "Rating Distribution": rating_distribution.round(2),
        "Sentiment Distribution": sentiment_distribution.round(2),
        "Category Distribution": category_distribution.round(2),
        "Subcategory Distribution": subcategory_distribution.round(2),
        "Detailing Distribution": detailing_distribution.round(2),
        "Positive Keywords and Bigrams": positive_keywords_bigrams,
        "Negative Keywords and Bigrams": negative_keywords_bigrams,
    }

# Função principal do Streamlit
def main():
    st.title("Análise Rápida de Reviews")

    uploaded_file = st.file_uploader("Faça upload do arquivo de reviews (.xlsx)", type="xlsx")
    if uploaded_file is not None:
        # Processar os dados
        with st.spinner("Processando dados..."):
            results = process_data(uploaded_file)

        # Mostrar métricas principais
        st.subheader("Resumo Geral")
        st.metric("Total de Reviews", results["Total Reviews"])
        st.metric("Média Geral de Rating", round(results["Average Rating"], 2))

        # Mostrar distribuições
        st.subheader("Distribuições")
        st.write("### Por Canal")
        st.dataframe(results["Channel Distribution"])

        fig = create_bar_chart(
            results["Channel Distribution"],
            x_col="Channel",
            y_col="Count",
            text_col="Percentage",
            title="Distribuição por Canal",
            labels={"Channel": "Canal", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig)

        st.write("### Por Status")
        st.dataframe(results["Status Distribution"])

        # Exibir Gráfico de Pizza para Distribuição por Status
        st.write("### Distribuição por Status")
        fig_status = create_pie_chart(
            results["Status Distribution"],
            values_col="Count",
            names_col="Status",
            title="Distribuição por Status",
            labels={"Status": "Status", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig_status)

        st.write("### Por Rating")
        st.dataframe(results["Rating Distribution"])

        # Exibir gráfico de Distribuição por Rating
        st.write("### Distribuição por Rating")
        fig_rating = create_bar_chart(
            results["Rating Distribution"],
            x_col="Rating",
            y_col="Count",
            text_col="Percentage",
            title="Distribuição por Rating",
            labels={"Rating": "Rating (⭐)", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig_rating)

        st.write("### Por Sentimento")
        st.dataframe(results["Sentiment Distribution"])

        fig = create_bar_chart(
            results["Sentiment Distribution"],
            x_col="Sentiment",
            y_col="Count",
            text_col="Percentage",
            title="Distribuição por Sentimento",
            labels={"Sentiment": "Sentimento", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig)

        st.write("### Por Categoria")
        st.dataframe(results["Category Distribution"])

        fig = create_bar_chart(
            results["Category Distribution"],
            x_col="Category",
            y_col="Count",
            text_col="Percentage",
            title="Distribuição por Categoria",
            labels={"Category": "Categoria", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig)

        st.write("### Por Subcategoria")
        st.dataframe(results["Subcategory Distribution"])

        fig = create_bar_chart(
            results["Subcategory Distribution"],
            x_col="Subcategory",
            y_col="Count",
            text_col="Percentage",
            title="Distribuição por Subcategoria",
            labels={"Subcategory": "Subcategoria", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig)

        st.write("### Por Detailing")
        st.dataframe(results["Detailing Distribution"])

        fig = create_bar_chart(
            results["Detailing Distribution"],
            x_col="Detailing",
            y_col="Count",
            text_col="Percentage",
            title="Distribuição por Detalhamentos",
            labels={"Detailing": "Detalhamentos", "Count": "Número de Reviews"}
        )
        st.plotly_chart(fig)



        # Mostrar análise textual
        # st.subheader("Análise Textual")
        # st.write("### Principais Palavras-Chave e Bigramas - Sentimentos Positivos")
        # st.write(results["Positive Keywords and Bigrams"])

        # st.write("### Principais Palavras-Chave e Bigramas - Sentimentos Negativos")
        # st.write(results["Negative Keywords and Bigrams"])

        # Exibir bigramas de forma legível
        st.write("### Principais keywords dos reviews positivos")
        if results["Positive Keywords and Bigrams"]:
            for word, freq in results["Positive Keywords and Bigrams"]:
                st.write(f"- **{word}** ({freq})")
        else:
            st.write("Nenhum dado disponível.")

        st.write("### Principais keywords dos reviews negativos")
        if results["Negative Keywords and Bigrams"]:
            for word, freq in results["Negative Keywords and Bigrams"]:
                st.write(f"- **{word}** ({freq})")
        else:
            st.write("Nenhum dado disponível.")

# Executar a aplicação
if __name__ == "__main__":
    main()
