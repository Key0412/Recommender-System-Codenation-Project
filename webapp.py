# Standard library imports
import base64
from zipfile import ZipFile

# Related third party imports
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Local application
from src.ReadFiles import ReadFiles
from src.LinearRegressionRecSys import LinearRegressionRecSys
from src.VisualizeLeads import VisualizeLeads
from src.GetDownloadLink import get_csv_download_link


st.set_option('deprecation.showfileUploaderEncoding', False)

# -- READ FILES --
@st.cache(show_spinner=False)
def reader():
    file_reader = ReadFiles()
    database, cluster_labels, original_market_df, portfolio2_snippet = file_reader.get_data()
    return database, cluster_labels, original_market_df, portfolio2_snippet
database, cluster_labels, original_market_df, portfolio2_snippet = reader()  

# -- MAIN --
def main():
    st.image("docs/header_photo.jpg", caption="Person holding a compass. Photo by Jamie Street on Unsplash.", use_column_width=True)
    
    # Title
    st.title("WebApp Leads Finder")
    st.subheader("Ache seu próximo parceiro de negócios")
    st.markdown("___")
    
    # Sobre
    st.header("Sobre")
    st.markdown("Este WebApp foi criado como projeto final do programa AceleraDev Data Science pela Codenation. Quer saber como funciona? [Obtenha mais informações na página do projeto](https://key0412.github.io/Recommender-System-Codenation-Project/).")
    st.markdown("___")
    
    # Como utilizar
    st.header("Como utilizar")
    st.markdown("O WebApp Leads Finder recomenda novos clientes para você! Envie seu portfolio através do campo mais abaixo e o sistema irá buscar as 50 melhores indicações com base no perfil de seus clientes.")
    st.markdown("O porfolio deve ser um arquivo .csv com os IDs de cada cliente e um header 'id', assim como no exemplo abaixo.")
    st.table(portfolio2_snippet.head(5))
    st.markdown("Caso deseje, você pode utilizar os portfolios exemplo disponíveis no link a seguir:")
    st.markdown("* [Acesse Porfolios Exemplo!](https://drive.google.com/drive/folders/1116lPSHfyPG2x5Z7VVv4ErAhhaX3pVMG?usp=sharing)")   
    st.markdown("Ou, pode obter uma amostra do dataset utilizado neste webapp:")
    st.markdown("* [Acesse a base de dados!](https://drive.google.com/drive/folders/1oVijJs-jOGJhbqNvMHxdTWpQyw8Dc860?usp=sharing)")  
    st.markdown("___")
    
    # Recomendações
    st.header("Recomendações")
    # File uploader:
    uploaded_portfolio = st.file_uploader("Envie seu portfolio de clientes (.csv):", type="csv")
        
    if uploaded_portfolio is not None:
        portfolio = pd.read_csv(uploaded_portfolio, usecols=["id"]) # read portfolio from updloaded file into a pandas df
        recsys = LinearRegressionRecSys(portfolio, database, cluster_labels) # create recsys object from portfolio dat
        with st.spinner('Treinando o modelo . . .'):
            recommendations = recsys.train_recommend(n_recommendations=50) # train models and recommend 50 leads
        st.success('Treinamento completo!')        
        visualizer = VisualizeLeads(recommendations["id"]) # instantiate VisualizeLeads
        st.subheader("Quem são seus novos clientes?")
        ranked_table = visualizer.create_table(original_market_df) # call method create_table()
        st.dataframe(ranked_table)
        st.markdown(get_csv_download_link(ranked_table), unsafe_allow_html=True) # Get download link for created table
        st.subheader("Quais suas características principais?")
        visualizer.create_barplots(original_market_df) # call method create_barplots() to plot features
        st.pyplot()        
    else:
        st.warning("**Oops! O portfolio não foi enviado!**")    
    st.markdown("___")
    
    # Autor
    st.header("Autor")
    st.markdown("Este WebApp foi criado em Python por **Klismam** Franciosi Pereira, estudante e entusiasta do campo de ciência de dados e engenheiro formado pela UFPR.")
    st.markdown("Entenda todo o processo e análise de dados na [página do projeto](https://key0412.github.io/Recommender-System-Codenation-Project/) e veja o [vídeo com uma explicação sobre as ideias por trás do sistema](https://www.youtube.com/watch?v=mPy3HNEKsns&feature=youtu.be).")
    st.markdown("Se tiver ideias de como melhorar este WebApp, se quiser criar um você mesmo, ou se tem uma ideia legal pra discutir, entre em contato!")
    st.subheader("Contatos")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/klismam-pereira/) | [Github](https://github.com/Key0412) | kp.franciosi@gmail.com")
    
if __name__ == "__main__":
    main()