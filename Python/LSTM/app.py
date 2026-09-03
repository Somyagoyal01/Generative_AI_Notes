import streamlit as st
import pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model(
        "nextword_model_fixed.keras",
        compile=False
    )


model = load_lstm_model()


# ----------------------------------------
# LOAD TOKENIZER
# ----------------------------------------
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# ----------------------------------------
# CREATE REVERSE WORD INDEX
# ----------------------------------------
reverse_index = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ----------------------------------------
# MAXIMUM SEQUENCE LENGTH
# ----------------------------------------
max_len = 44


# ----------------------------------------
# GENERATE TEXT
# ----------------------------------------
def generate_text(seed_text, num_words=10):

    text = seed_text

    for _ in range(num_words):

        # Convert input text into token sequence
        sequence = tokenizer.texts_to_sequences([text])[0]

        # Pad sequence to required length
        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="pre"
        )

        # Predict next word probabilities
        prediction = model.predict(
            padded_sequence,
            verbose=0
        )

        # Get index with highest probability
        predicted_index = np.argmax(prediction)

        # Convert index back to word
        next_word = reverse_index.get(predicted_index, "")

        # Stop if no word is found
        if next_word == "":
            break

        # Add predicted word to text
        text += " " + next_word

    return text


# ----------------------------------------
# STREAMLIT USER INTERFACE
# ----------------------------------------
st.title("Next Word Prediction with Deep Learning")

st.write(
    "Enter some starting text and let the LSTM predict the next words."
)


seed_text = st.text_input(
    "Enter a Starting Text:",
    "Hello"
)


num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=20,
    value=10
)


if st.button("Generate"):

    if seed_text.strip():

        result = generate_text(
            seed_text,
            num_words
        )

        st.success(result)

    else:
        st.warning("Please enter some starting text.")