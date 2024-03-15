import streamlit as st
import pandas as pd
import plotly.express as px


# charger les données depuis un fichier CSV
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, sep=';')


# afficher la segmentation des clients
def segmentation_des_clients(df):
    st.subheader("Nous avons catégorisé les clients en plusieurs types, à travers du clustering")

    st.image("C:/Users/SYLVAIN/Downloads/Data Olist/Images/segmengtation_client.png")


# afficher la page d'accueil
def page_accueil():
    st.title("Analyse des clients Olist")
    st.image("C:/Users/SYLVAIN/Downloads/Data Olist/Images/olist.png", width=400)
    st.write("""
    Nous fournissons à travers le dashboard, une vision claire des différents types de clients, de leur comportement d'achat et des tendances importantes à prendre en compte pour les campagnes de communication ! 
    """)


# afficher l'origine du chiffre d'affaires
def sentiment_analysis(data):
    st.subheader("Nous avons classé les commentaires sous trois groupes : Positif, Négatif et Neutre à travers la méthode du sentiment analysis")
    st.image("C:/Users/SYLVAIN/Downloads/Data Olist/Images/sentiment_analysis.png")


# afficher les clients payant le plus de frais de livraison
def paiment_frais_livraison(data):
    st.subheader("Nous avons identifé les clients qui payent le plus de frais de livraison par rapport à la classification. Seul le client fidèle bénéficie des avantages de la livraison")

    st.image("C:/Users/SYLVAIN/Downloads/Data Olist/Images/plus_de_frais_de_livraison.png")
   

# afficher les clients payant le plus de frais de livraison
def produits_plus_vendus(data):
    st.subheader("Nous avons représenté les catégories des produits les plus vendus avec le chiffre d'affaire généré")

    st.image("C:/Users/SYLVAIN/Downloads/Data Olist/Images/CA.png")


# afficher les moyens de paiement utilisés par les clients
def mode_paiement_plus_utilise(data):
    st.subheader("Nous avons représenté les catégories des produits les plus vendus avec le chiffre d'affaires (C.A) généré")

    st.image("C:/Users/SYLVAIN/Downloads/Data Olist/Images/mode_paimnt.png")


# Charger les données
df = load_data('C:/Users/SYLVAIN/Downloads/Data Olist/Data/df_user.csv')

# Liste des options de graphiques
menu_options = {
    "Accueil": page_accueil,
    "Segmentation des clients": segmentation_des_clients,
    "Origine du chiffre d'affaires": sentiment_analysis,
    "Clients payant le plus de frais de livraison": paiment_frais_livraison,
    "Catégories de produits les plus vendues": produits_plus_vendus,
    "Moyens de paiement utilisés par les clients": mode_paiement_plus_utilise
}


# Afficher la barre de menu
st.sidebar.title("Menu")
selected_page = st.sidebar.radio("Sélectionnez une option", list(menu_options.keys()))


# Exécuter la fonction correspondante à l'option sélectionnée dans le menu
if selected_page == "Accueil":
    menu_options[selected_page]()  # Pas besoin de passer de DataFrame en argument pour la page d'accueil
else:
    menu_options[selected_page](df)  # Passer le DataFrame en argument pour les autres pages
