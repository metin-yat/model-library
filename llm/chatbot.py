import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
# Install pip install llama-index-llms-ollama along witht the llama_index
from llama_index.llms.ollama import Ollama
from langchain_community.chat_models import ChatOllama


def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e


logging.basicConfig(level=logging.INFO)

if 'messages' not in st.session_state:
    st.session_state.messages = []

menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style
# Main title of streamlit application
main_title_cfg = """<div>
                        <h1 style="color:#00000; text-align:center; font-size:40px; margin-top:-50px;
                                                font-family: 'Archivo', sans-serif; margin-bottom:20px;">
                            Chat with Fine-tuned Llama Model
                        </h1>
                    </div>"""
st.markdown(menu_style_cfg, unsafe_allow_html=True)
st.markdown(main_title_cfg, unsafe_allow_html=True)
logging.info("App started")

#model = st.sidebar.selectbox("Choose a model", ["mymodel", "llama3.1 8b", "phi3", "mistral"])
model = "finetunedllama38binstruct4bit:latest"
logging.info(f"Model selected: {model}")

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    logging.info(f"User input: {prompt}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            start_time = time.time()
            logging.info("Generating response")

            with st.spinner("Writing..."):
                try:
                    messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                    response_message = stream_chat(model, messages)
                    duration = time.time() - start_time
                    response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                    st.session_state.messages.append({"role": "assistant", "content": response_message_with_duration})
                    st.write(f"Duration: {duration:.2f} seconds")
                    logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": str(e)})
                    st.error("An error occurred while generating the response.")
                    logging.error(f"Error: {str(e)}")