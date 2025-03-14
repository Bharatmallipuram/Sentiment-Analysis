import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import joblib
import numpy as np

# Loading the model and TF-IDF vectorizer
model = joblib.load("model.pkl")
tf_idf_vector = joblib.load("tfidf.pkl")

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
sw = set(stopwords.words("english"))

def predict(review):
    # Cleaning the review text
    cleaned_review = re.sub("<.*?>", "", review)
    cleaned_review = re.sub(r'[^\w\s]', "", cleaned_review)
    cleaned_review = cleaned_review.lower()
    
    # Tokenizing, removing stopwords, and stemming
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in sw]
    stemmed_review = [stemmer.stem(word) for word in filtered_review]
    
    # Transforming the review text using TF-IDF vectorizer
    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])
    
    # Predicting the sentiment
    sentiment_predict = model.predict(tfidf_review)
    
    # Interpreting the prediction result
    sentiment_class = np.argmax(sentiment_predict, axis=1)  # Get the class with the highest probability
    
    if sentiment_class == 1:  # Assuming 1 is Positive
        return "Positive"
    else:
        return "Negative"

# Example usage
user_input = input("Enter your text here: ")
predicted_sentiment = predict(user_input)
print("Predicted Sentiment:", predicted_sentiment)

# If you want to use Streamlit for a web UI, uncomment the following lines
# import streamlit as st
# st.title("Sentiment Analysis")
# review_to_predict = st.text_area("Enter your text here:")
# if st.button('Predict Sentiment'):
#     predicted_sentiment = predict(review_to_predict)
#     st.write("Predicted Sentiment:", predicted_sentiment)



