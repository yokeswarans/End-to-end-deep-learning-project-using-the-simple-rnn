"""import numpy as np 
import tensorflow as tf  
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

##load imdb
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

#Load the pretrained model with RELU activation
model=load_model('Simple_RNN_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


#prediction function
def predict(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    return sentiment,prediction[0][0]


import streamlit as st  
st.title("IMDB Movie Review Sentimental Analysis")
st.write("Enter a Movie review to classify it as positive or negative")


#User Inpur
user_input=st.text_area("Movie_review")


if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    #make prediction
    prediction=model.predict(preprocess_input)
    sentiments='Positive' if prediction[0][0]>0.5 else 'Negative'


    #Display the result
    st.write("The sentiment of the review is: ",sentiments)
    st.write("Prediction score",prediction[0][0])
else:
    st.write("please enter a movie review")"""




import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load IMDB word index and reverse mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pretrained model
model = load_model('Simple_RNN_imdb.h5')

# Helper to decode encoded reviews (for demo/sample purposes)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocessing: text → padded sequence
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # OOV=2
    # Clip out-of-range tokens
    encoded_review = [min(i, 9999) for i in encoded_review]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

'''# Sample reviews from IMDB test set for users
(_, _), (X_test, y_test) = imdb.load_data(num_words=10000)
sample_reviews = [
    (decode_review(X_test[0]), y_test[0]),
    (decode_review(X_test[1]), y_test[1]),
    (decode_review(X_test[2]), y_test[2])
]

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")
st.markdown("<h1 style='text-align:center; color: #FF4B4B;'>🎬 IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.write("Enter a movie review below or pick one of the sample IMDB reviews to see if it's **Positive** or **Negative**.")

st.subheader("📚 Sample Reviews")
cols = st.columns(3)
for idx, (review, label) in enumerate(sample_reviews):
    label_text = "✅ Positive" if label == 1 else "❌ Negative"
    if cols[idx].button(f"Sample {idx+1} ({label_text})"):
        st.session_state["user_input"] = review'''

# Sample reviews for testing
st.subheader("📌 Sample Reviews")
sample_positive = [
    "An absolute masterpiece. The performances were stunning and the story kept me hooked from start to finish.",
    "Beautiful cinematography and a moving soundtrack. Easily one of the best films I've seen in years.",
    "I loved every second of it. The plot twists were unexpected but made perfect sense."
]
sample_negative = [
    "A complete waste of time. The acting was wooden and the plot made no sense at all.",
    "Poor pacing and predictable storyline. I nearly fell asleep halfway through.",
    "Overhyped and underwhelming. I regret spending money on this."
]

# Display in two columns
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Positive**")
    for review in sample_positive:
        st.write(f"✅ {review}")

with col2:
    st.markdown("**Negative**")
    for review in sample_negative:
        st.write(f"❌ {review}")

# Review input
user_input = st.text_area("📝 Movie Review", value=st.session_state.get("user_input", ""), height=150)

# Predict button
if st.button('🔍 Classify Sentiment'):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)

        # Color-coded sentiment display
        if sentiment == "Positive":
            st.markdown(f"<h3 style='color:green;'>✅ Sentiment: Positive</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red;'>❌ Sentiment: Negative</h3>", unsafe_allow_html=True)

        # Confidence bar
        st.progress(float(score))
        st.write(f"**Prediction Confidence:** {score:.2%}")

    else:
        st.warning("⚠ Please enter a review or select a sample above.")
else:
    st.info("ℹ Enter a review or choose a sample to start.")
