import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_seq_len = 20  # same as training

def generate_text(seed_text):
    for _ in range(20):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        seed_text += " " + output_word

    return seed_text

interface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="LSTM Text Generator"
)

