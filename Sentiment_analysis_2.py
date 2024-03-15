
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import pandas as pd
import plotly.graph_objects as go

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Définition des listes de mots positifs et négatifs
positive_words = [
    "feliz", "alegria", "aatisfeito", "agradável", "excelente",
    "Rápido", "eficiente", "confiável", "confortável", "prático",
    "oportuno", "simples", "amigável", "util", "seguro",
    "moderno", "inovador", "flexível", "encantador", "eficaz"
]

negative_words = [
    "triste", "desapontado", "insatisfeito", "desagradável", "ruim",
    "lento", "ineficiente", "inconfiável", "desconfortável", "complicado",
    "tardio", "difícil", "hostil", "inútil", "inseguro",
    "antiquado", "obsoleto", "rígido", "desagradável", "ineficaz"
]

# Définition des poids pour les mots positifs et négatifs
positive_weight = 1
negative_weight = -1

df = pd.read_csv('order_reviews_dataset.csv', sep = ';')

df_comments = df[['review_comment_message']]

df_comments = df_comments.dropna()


# Fonction de prétraitement des commentaires
def preprocess(comment, language='spanish'):
    # Tokenization
    tokens = word_tokenize(comment)
    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    # Suppression des mots vides
    if language == 'portuguese':
        stop_words = set(stopwords.words('spanish'))
    elif language == 'spanish':
        stop_words = set(stopwords.words('spanish'))
    else:
        stop_words = set(stopwords.words('spanish'))
    filtered_tokens = [token for token in lemmatized_tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens


df_comments['review_comment_message'] = df_comments['review_comment_message'].apply(preprocess)
negative_words = [preprocess(word) for word in negative_words]
negative_words
positive_words = [preprocess(word) for word in positive_words]
positive_words
df_comments


# Fonction pour calculer le score de sentiment
def calculate_sentiment_score(tokens):
    sentence = ' '.join(tokens)  # Reconstituer la phrase à partir des tokens
    blob = TextBlob(sentence)
    sentiment_score = blob.sentiment.polarity

    # Vérifiez les mots positifs et négatifs dans la phrase
    for word in blob.words:
        if word in positive_words:
            sentiment_score += 0.2  # Ajouter un poids positif
        elif word in negative_words:
            sentiment_score -= 0.2  # Ajouter un poids négatif
    
    # Limiter le score de sentiment entre -1.0 et 1.0
    sentiment_score = max(-1.0, min(sentiment_score, 1.0))
    
    return sentiment_score


# Application de la fonction de calcul du score de sentiment aux commentaires
df_comments['sentiment_score'] = df_comments['review_comment_message'].apply(calculate_sentiment_score)
print(df_comments.head())
df_comments
df_comments['sentiment_score'].unique()


def determine_sentiment(score):
    if score > 0:
        return 'positif'
    elif score < 0:
        return 'négatif'
    else:
        return 'neutre'


# Application de la fonction pour déterminer le sentiment à la colonne 'sentiment'
df_comments['sentiment'] = df_comments['sentiment_score'].apply(determine_sentiment)

df_comments
df_comments['sentiment'].unique()

# Compter le nombre de commentaires par sentiment
sentiment_counts = df_comments['sentiment'].value_counts()

# Création du graphique à barres
fig = go.Figure(data=[
    go.Bar(name='Positif', x=['Positif'], y=[sentiment_counts.get('positif', 0)], marker=dict(color='green')),
    go.Bar(name='Négatif', x=['Négatif'], y=[sentiment_counts.get('négatif', 0)], marker=dict(color='red')),
    go.Bar(name='Neutre', x=['Neutre'], y=[sentiment_counts.get('neutre', 0)], marker=dict(color='blue'))

])

# Mise en forme du titre et des axes
fig.update_layout(title='Nombre de commentaires par sentiment',
                  xaxis_title='Sentiment',
                  yaxis_title='Nombre de commentaires')


# Affichage du graphique
fig.show()
