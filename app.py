import json
import random
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

intents = json.load(open('intents.json'))
tags = []
prompts = []

for intent in intents['intents']:
    for p in intent['prompt']:
        prompts.append(p)
        tags.append(intent['tag'])

vector = TfidfVectorizer()
prompt_scaled = vector.fit_transform(prompts)

# building model
Bot = LogisticRegression(max_iter=100000)
Bot.fit(prompt_scaled, tags)

# testing the model
def ChatBot(input_message):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]

    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            response = random.choice(intent['response'])
            return response

st.markdown(
    """
    <h1 style='text-align: center; color: blue;'>Welcome to Australia</h1>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prepopulated questions
prepopulated_questions = [
    "What is the capital of Australia?",
    "Which are major cities in Australia?",
    "How is the weather in Australia?",
    "What is the currency of Australia?",
    "Which is official language of Australia?",
    "Which is national sport of Australia?",
    "What is population of Australia?",
    "Which is national animal of Australia?",
    "Which are international airports of Australia?",
    "hi",
    "bye",
    "who are you?"
]

# Create buttons for prepopulated questions
cols = st.columns(3) # create 3 columns for better layout
for i, question in enumerate(prepopulated_questions):
    if cols[i % 3].button(question): # place buttons in columns
        p = question
        st.chat_message("user").markdown(p)
        st.session_state.messages.append({"role": "user", "content": p})

        response = f"Kristy: " + ChatBot(p)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# React to user input


if p := st.chat_input("Enter your message here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(p)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": p})

    response = f"Kristy: " + ChatBot(p)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})