import streamlit as st
import numpy as np
import spacy
import re
import pandas as pd
# from tensorflow.python.keras.models import load_model
# from keras.models import load_model
from nltk.corpus import stopwords
from collections import Counter
from datetime import datetime
import sklearn
import joblib
from pathlib import Path
import nltk
import datetime


model = joblib.load('./Random_Forest.joblib')

# Example function to predict using the loaded model
def predict_tweet_likes(X):
    # Preprocess the input tweet as required by the model
    # For example, tokenize the tweet, convert to lowercase, remove stop words, etc.
    
    # Use the loaded model to make predictions
    prediction = model.predict(X)
    
    # Return the prediction
    return prediction[0]  # Assuming model.predict() returns a list of predictions

# Example of preprocessing function (you need to define your own preprocessing logic)
# def preprocess(tweet):
#     # Your preprocessing logic here
#     # Tokenization, lowercase conversion, stop word removal, etc.
#     return preprocessed_tweet

# Example usage
# tweet = "This is a great tweet!"
# prediction = predict_tweet_sentiment(tweet)
# print("Prediction:", prediction)


#removing noise from the tweet data like non-alphanumeric and hashes
def contains_https_link(tweet):
    return bool(re.search(r'https://', tweet))

def clean_tweet(tweet):
    all_text  = re.sub("#\S*\s", "", tweet)
    all_text  = re.sub("W+", "", all_text)
    all_text  = re.sub("@\S*\s", "", all_text)
    all_text = re.sub(r'https?://\S+', '', all_text)
    all_text = re.sub(r'[^\x00-\x7F]+', '', all_text)
    return all_text

def standard_time(timestamp):
    hour = timestamp.hour
    minute = timestamp.minute
    time_numeric = (hour * 3600) + (minute * 60)
    return time_numeric

def label_to_likes(label):
    if label == 4:
        return '10000-30000'
    if label == 3:
        return '5000-10000'
    if label == 2:
        return '1000-5000'
    if label == 1:
        return '100-1000'
    else:
        return '0-100'


nltk.download('stopwords')
stopwords1 = stopwords.words('english')

en = spacy.load("en_core_web_md")
stopwords2 = en.Defaults.stop_words
              
stop_words = stopwords1 + list(stopwords2)
# print(stop_words)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def calculate_word_scores(sentences):
    word_freq = Counter()
    word_degree = Counter()
    for sentence in sentences:
        word_list = re.findall(r'\w+', sentence)
        word_list = [word for word in word_list if len(word) > 1]  # Filter out single-character words
        word_freq.update(word_list)
        for word in word_list:
            word_degree[word] += len(word_list) - 1  # Increment the degree by the count of other words in the sentence
   
    word_scores = Counter()
    for word in word_freq:
        word_scores[word] = word_degree[word] / word_freq[word]
    return word_scores

def calculate_phrase_scores(sentences, word_scores):
    phrase_scores = Counter()
    for sentence in sentences:
        phrase_list = re.findall(r'\w+', sentence)
        phrase_list = [phrase for phrase in phrase_list if len(phrase) > 1]  # Filter out single-word phrases
        phrase_score = sum(word_scores[word] for word in phrase_list)
        phrase_scores[' '.join(phrase_list)] = phrase_score
    return phrase_scores

def extract_keywords(text, num_keywords=5):
    split_pattern = r'[.!?]|(?:\s|^)(?:{})\b'.format('|'.join(map(re.escape, stop_words)))
    text = preprocess_text(text)
    sentences = re.split(split_pattern, text)
    sentences = [sentence for sentence in sentences if sentence.strip()]
    word_scores = calculate_word_scores(sentences)
    phrase_scores = calculate_phrase_scores(sentences, word_scores)
    keywords = phrase_scores.most_common(num_keywords)
    most_words = word_scores.most_common(num_keywords)
    return keywords, most_words

def keyword_to_embeddings(top_keywords):
    word_embeddings = []
    for keyword in top_keywords:
        token = en(keyword[0])
        if token.has_vector:
            word_embeddings.append(token.vector)
        else:
            word_embeddings.append(np.zeros(96))
   
    while len(word_embeddings) < 5:
        word_embeddings.append(np.zeros(96))
       
    return word_embeddings


# Function to predict likes based on input features
def predict_likes(tweet, media_flag, time, followers):
    X_text_embeddings = []
    X_followers_embeddings = []
    X_date_embeddings = []

    _, words = extract_keywords(tweet)
    content = ' '.join(word[0] for word in words)
    doc = en(content)
    content_embedding = doc.vector

    X_text_embeddings.append(content_embedding)    
    X_followers_embeddings.append(followers)     
    X_date_embeddings.append(standard_time(time))  
    media_flag |= contains_https_link(tweet)
    # tweet = clean_tweet(tweet)
    # time_num = standard_time(time)
    # _,tweet_top_keywords = extract_keywords(tweet)
    # embeddings = keyword_to_embeddings(tweet_top_keywords)

    # X_followers_embeddings.append(followers)
    # X_text_embeddings.append(embeddings)
    # X_date_embeddings.append(time_num)

    X_text_embeddings_df = pd.DataFrame(X_text_embeddings)
    X_followers_embeddings_df = pd.DataFrame(X_followers_embeddings)
    X_date_embeddings_df = pd.DataFrame(X_date_embeddings)
    X_combined3 = pd.concat([ X_text_embeddings_df, X_followers_embeddings_df, X_date_embeddings_df], axis=1)

    features = X_combined3
    # print(features.shape)
    # print(features[3].shape)
    label = predict_tweet_likes(features)                                                                                                           - 1                  
    likes = label_to_likes(label)
    return likes, words

print("1")
# Streamlit UI
st.markdown("<div style='width: 100%;'><img src='https://d3lzcn6mbbadaf.cloudfront.net/media/details/ANI-20230724104554.jpg' style='display: block; margin-left: auto; margin-right: auto;' width=200 alt='X_logo' </div>", unsafe_allow_html=True)
# st.image('https://d3lzcn6mbbadaf.cloudfront.net/media/details/ANI-20230724104554.jpg', width=100, caption="Twitter Logo", use_column_width=True, output_format="SVG", align='center')

st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 50px;'> Tweet Likes Prediction </h1>", unsafe_allow_html=True)
print("2")
# Input fields for user
tweet_text = st.text_area('Enter your tweet here:')
tweet_time = st.time_input('Select Time of Tweet:', datetime.time(12, 00))
media_flag = st.checkbox('Does the tweet contain embedded media?')
num_followers = st.number_input('Enter the number of followers:', min_value=0)

columns = st.columns((1,1,1))

# Button to trigger prediction
if columns[1].button('Predict Likes'):
    if tweet_text.strip() == '':
        st.warning('Please fill in all fields.')
    else:
        likes_prediction, top_keywords = predict_likes(tweet_text, media_flag, tweet_time, num_followers)
        st.success(f'Top Keywords with their score: {top_keywords}')
        st.success(f'Predicted likes for the tweet: {likes_prediction}')