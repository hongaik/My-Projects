import pickle
import numpy as np
import pandas as pd
import streamlit as st
import re
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
lr = pickle.load(open('trained_lr.sav', 'rb'))
tfidf = pickle.load(open('trained_tfidf.sav', 'rb'))

def cleaner(text):
    
    text = text.lower()
    
    text = re.sub(r'[^0-9a-zA-Z\s]', '', text)
    
    text = re.sub(r'\s\s+', ' ', text)
    
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(i) for i in text.split()]) #lemmatize
    
    return text

def make_predictions(input_text):
    
    input_text = cleaner(input_text)
    
    dtm = tfidf.transform([input_text]).todense()

    padding = np.array((0,0)).reshape(1,-1)

    dtm = pd.DataFrame(np.hstack((dtm, padding)))
    
    if dtm.values.sum() == 0:
        return 'No words detected! Try another string of words.'
    
    prediction = lr.predict(dtm)
    proba = lr.predict_proba(dtm)
    proba = round(100*max(proba[0]))
    
    if prediction == 1:
        return 'With ' + str(proba) + '% confidence, this is a submission belonging to r/Zoom.'
    else:
        return 'With ' + str(proba) + '% confidence, this is a submission belonging to r/MicrosoftTeams.'       

def main():
    
    result = ''
    
    st.title('Reddit Classification Web App')
    
    st.markdown('The underlying model was trained on ~14,000 sub-reddits from r/Zoom and r/MicrosoftTeams, with the goal of predicting the sub-reddit given a string of words (submission)')
    st.markdown('The model is only able to output 2 possible results!')
    
    text = st.text_input('Type your content here!')
    
    if st.button('Click for predictions!'):
        result = make_predictions(text)
    
    st.success(result)
    
if __name__ == '__main__':
    main()