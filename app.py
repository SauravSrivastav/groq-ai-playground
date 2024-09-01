import os
import streamlit as st
from typing import Generator
from groq import Groq
import json
from datetime import datetime
from fpdf import FPDF
import base64
import torch
from transformers import pipeline
from datasets import load_dataset
from torch.nn.attention import SDPBackend, sdpa_kernel

# Set page configuration
st.set_page_config(page_icon="🤖", layout="wide", page_title="Groq AI Playground")

# Page title and description
st.title("Groq AI Playground")
st.markdown("""
    Welcome to the Groq AI Playground! Explore multiple language models and experience the power of Groq's API.
    Select a model, adjust parameters, and start interacting with advanced AI models for text generation.
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Define model details
models = {
    "gemma2-9b-it": {"name": "Gemma 2 9B", "tokens": 8192, "developer": "Google"},
    "gemma-7b-it": {"name": "Gemma 7B", "tokens": 8192, "developer": "Google"},
    "llama3-groq-70b-8192-tool-use-preview": {"name": "Llama 3 Groq 70B Tool Use (Preview)", "tokens": 8192, "developer": "Groq"},
    "llama3-groq-8b-8192-tool-use-preview": {"name": "Llama 3 Groq 8B Tool Use (Preview)", "tokens": 8192, "developer": "Groq"},
    "llama-3.1-70b-versatile": {"name": "Llama 3.1 70B (Preview)", "tokens": 8192, "developer": "Meta"},
    "llama-3.1-8b-instant": {"name": "Llama 3.1 8B (Preview)", "tokens": 8192, "developer": "Meta"},
    "llama-guard-3-8b": {"name": "Llama Guard 3 8B", "tokens": 8192, "developer": "Meta"},
    "llama3-70b-8192": {"name": "Meta Llama 3 70B", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "Meta Llama 3 8B", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral"},
}

# Function to convert chat history to plain text
def chat_to_text(messages):
    text = "Groq AI Playground - Chat Export\n\n"
    for msg in messages:
        text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    return text

# Function to convert chat history to PDF
def chat_to_pdf(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Groq AI Playground - Chat Export", ln=1, align='C')
    for msg in messages:
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(200, 10, txt=f"{msg['role'].capitalize()}:", ln=1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt=msg['content'].encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# Function to create a download link
def get_download_link(file_content, file_name, file_format):
    if file_format in ['txt', 'json']:
        b64 = base64.b64encode(file_content.encode('utf-8')).decode()
    else:  # pdf
        b64 = base64.b64encode(file_content).decode()

    mime_types = {
        'txt': 'text/plain',
        'json': 'application/json',
        'pdf': 'application/pdf'
    }
    mime = mime_types.get(file_format, 'application/octet-stream')

    href = f'<a href="data:{mime};base64,{b64}" download="{file_name}">Download {file_format.upper()} File</a>'
    return href

# Sidebar for configuration
with st.sidebar:
    st.markdown("### Configuration")

    # API Key input
    st.markdown("#### API Key")
    api_key = st.text_input("Enter your Groq API Key:", type="password", placeholder="Your API Key")
    if api_key:
        st.session_state.api_key = api_key

    # Add link to get API key
    st.markdown("[Get your Groq API key here](https://console.groq.com/keys)")

    # Model selection
    st.markdown("#### Model Selection")
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=0,
    )

    # Display model information
    st.markdown("#### Model Information")
    st.markdown(f"""
    **Name:** {models[model_option]['name']}
    **Max Tokens:** {models[model_option]['tokens']}
    **Developer:** {models[model_option]['developer']}
    """)

    # Max tokens slider
    st.markdown("#### Parameters")
    max_tokens_range = models[model_option]["tokens"]
    min_tokens_value = 512 if max_tokens_range > 512 else 1
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=min_tokens_value,
        max_value=max_tokens_range,
        value=min(4096, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens for the model's response. Max: {max_tokens_range}",
    )

    # Temperature slider
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust the randomness of the model's responses. Higher values make output more random.",
    )

    # Clear chat button
    st.markdown("#### Actions")
    if st.button("Clear Chat"):
        st.session_state.messages = []

    # Export chat functionality
    st.markdown("#### Export Chat")
    export_format = st.selectbox(
        "Choose export format:",
        options=["JSON", "TXT", "PDF"],
        index=0,
    )

    if st.button("Export Chat"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if export_format == "JSON":
                chat_export = {
                    "model": st.session_state.selected_model,
                    "timestamp": timestamp,
                    "messages": st.session_state.messages
                }
                file_content = json.dumps(chat_export, indent=2)
                file_name = f"groq_chat_export_{timestamp}.json"
            elif export_format == "TXT":
                file_content = chat_to_text(st.session_state.messages)
                file_name = f"groq_chat_export_{timestamp}.txt"
            else:  # PDF
                file_content = chat_to_pdf(st.session_state.messages)
                file_name = f"groq_chat_export_{timestamp}.pdf"

            st.markdown(
                get_download_link(file_content, file_name, export_format.lower()),
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"An error occurred during export: {str(e)}")

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Main interaction area
st.markdown("### Interact with AI")
interaction_type = st.radio("Choose interaction type:", ("Text Chat",))

if interaction_type == "Text Chat":
    # Chat input for text-based models
    if prompt := st.chat_input("Enter your prompt here..."):
        if not st.session_state.api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            # Create Groq client
            def create_groq_client():
                try:
                    return Groq(api_key=st.session_state.api_key)
                except Exception as e:
                    st.error(f"Failed to create Groq client: {str(e)}", icon="❌")
                    return None

            client = create_groq_client()
            if client:
                # Fetch response from Groq API
                try:
                    chat_completion = client.chat.completions.create(
                        model=model_option,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                    # Use the generator function to collect responses
                    response_chunks = []
                    with st.chat_message("assistant", avatar="🤖"):
                        message_placeholder = st.empty()
                        for chunk in generate_chat_responses(chat_completion):
                            response_chunks.append(chunk)
                            message_placeholder.markdown(''.join(response_chunks) + "▌")
                        full_response = ''.join(response_chunks)
                        message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}", icon="❌")

# Add footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Note:</strong> This is a demo application showcasing the capabilities of various AI models through the Groq API.</p>
    <p>Developed by <a href="https://github.com/SauravSrivastav">Saurav Srivastav</a></p>
</div>
""", unsafe_allow_html=True)
