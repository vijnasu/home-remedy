import streamlit as st
import torch
import json
import random
import textwrap

from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# Load intents
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))  # Load to CPU
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "HomeRemedy"

# Function to wrap text
def wrap_text(text, width=50):
    return textwrap.fill(text, width)

# Streamlit app
st.title("Chat with HomeRemedy Bot")
st.write("Type 'quit' to end the conversation.")

# Chat input
user_input = st.text_input("You: ")

if user_input.lower() == "quit":
    st.stop()

if st.button("Send"):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                wrapped_response = wrap_text(f"{bot_name}: {response}")
                st.text(wrapped_response)
    else:
        wrapped_response = wrap_text(f"{bot_name}: I do not understand...")
        st.text(wrapped_response)
