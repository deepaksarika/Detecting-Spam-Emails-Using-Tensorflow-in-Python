import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Download NLTK resources
nltk.download('stopwords')

# Functions for Data Processing and Model Building
def load_data(file_path):
    return pd.read_csv(file_path)

def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = [word.lower() for word in str(text).split() if word.lower() not in stop_words]
    return " ".join(imp_words)

def preprocess_data(data):
    # Remove punctuations
    data['text'] = data['text'].apply(lambda x: remove_punctuations(x))
    # Remove stopwords
    data['text'] = data['text'].apply(lambda x: remove_stopwords(x))
    return data

def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['text'])
    plt.figure(figsize=(7, 7))
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400, collocations=False).generate(email_corpus)
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    plt.axis('off')
    st.pyplot()

def build_model(train_X, train_Y, test_X, test_Y):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_X)
    
    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)
    
    max_len = 100
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len))
    model.add(tf.keras.layers.LSTM(16))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
    lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)
    
    history = model.fit(train_sequences, train_Y, validation_data=(test_sequences, test_Y), epochs=20, batch_size=32, callbacks=[lr, es])
    
    return model, history

# Streamlit UI
def main():
    st.title('Spam Classification App')
    
    # Load data
    st.sidebar.header('Data')
    file_path = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    
    if file_path is not None:
        data = load_data(file_path)
        st.dataframe(data.head())
        
        # Preprocess data
        data = preprocess_data(data)
        
        # Plot WordClouds
        st.sidebar.header('WordClouds')
        st.subheader('WordCloud for Non-Spam Emails')
        plot_word_cloud(data[data['spam'] == 0], 'Non-Spam')
        
        st.subheader('WordCloud for Spam Emails')
        plot_word_cloud(data[data['spam'] == 1], 'Spam')
        
        # Train-test split
        st.sidebar.header('Train-Test Split')
        train_X, test_X, train_Y, test_Y = train_test_split(data['text'], data['spam'], test_size=0.2, random_state=42)
        
        # Build model
        st.sidebar.header('Model Building')
        model, history = build_model(train_X, train_Y, test_X, test_Y)
        
        # Model evaluation
        st.sidebar.header('Model Evaluation')
        test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
        st.write('Test Loss:', test_loss)
        st.write('Test Accuracy:', test_accuracy)
        
        # Plot accuracy
        st.header('Model Accuracy')
        st.line_chart(history.history['accuracy'])
        st.line_chart(history.history['val_accuracy'])

if __name__ == '__main__':
    main()
