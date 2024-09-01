# groq-ai-playground 🤖

groq-ai-playground is an interactive platform to explore and experiment with various AI models powered by Groq. Customize parameters, chat with AI, and export your conversations in multiple formats.

## Features ✨

- **Fast and Efficient**: Powered by Groq's LPU for low latency and high performance.
- **Customizable**: Modify system prompts and choose from various models.
- **Interactive UI**: Built with Streamlit for an intuitive user interface.
- **Export Options**: Export chat history in JSON, TXT, or PDF formats.

## Installation 🛠️

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SauravSrivastav/groq-ai-playground.git
    cd groq-ai-playground
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your Groq API Key**:
    - Sign up on the [Groq website](https://console.groq.com/keys) and generate an API key.
    - Set the API key as an environment variable:
        ```bash
        export GROQ_API_KEY=your_groq_api_key
        ```

## Running the Application 🚀

1. **Start the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Open your browser** and navigate to `http://localhost:8501` to interact with the AI models.

## Usage Instructions 📖

1. **Enter your Groq API Key** in the sidebar.
2. **Choose a model** from the dropdown menu.
3. **Adjust parameters** such as max tokens and temperature.
4. **Ask a question** in the text input box and get responses from the AI.

## App Screenshots 📸

![App Screenshot 1](https://github.com/SauravSrivastav/groq-ai-playground/blob/main/data/1.png)   
![App Screenshot 2](https://github.com/SauravSrivastav/groq-ai-playground/blob/main/data/2.png)  

## Detailed Code Explanation 🧩

### Main Application (`app.py`)

#### Imports and Setup
```python
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
```

- Import necessary libraries and modules.
- Set up the main function for the Streamlit application.

#### Sidebar Configuration
```python
with st.sidebar:
    st.markdown("### Configuration")
    api_key = st.text_input("Enter your Groq API Key:", type="password", placeholder="Your API Key")
    if api_key:
        st.session_state.api_key = api_key
    st.markdown("[Get your Groq API key here](https://console.groq.com/keys)")
    model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0)
    st.markdown(f"**Name:** {models[model_option]['name']} **Max Tokens:** {models[model_option]['tokens']} **Developer:** {models[model_option]['developer']}")
    max_tokens = st.slider("Max Tokens:", min_value=512, max_value=models[model_option]["tokens"], value=4096, step=512)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    if st.button("Clear Chat"):
        st.session_state.messages = []
    export_format = st.selectbox("Choose export format:", options=["JSON", "TXT", "PDF"], index=0)
    if st.button("Export Chat"):
        # Export logic here
```

- Provide instructions for obtaining the Groq API key.
- Input field for the user to enter their API key.
- Model selection and parameter adjustment.

#### Chat Interaction
```python
if interaction_type == "Text Chat":
    if prompt := st.chat_input("Enter your prompt here..."):
        if not st.session_state.api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
            client = create_groq_client()
            if client:
                try:
                    chat_completion = client.chat.completions.create(
                        model=model_option,
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
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
```

- Initialize session state for chat history.
- Handle user input and display chatbot responses.

## Contributing 🤝

Contributions are welcome! If you'd like to improve groq-ai-playground, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact Us

Have questions or suggestions? Reach out to us:

- 📧 Email: [Sauravsrivastav2205@gmail.com](mailto:Sauravsrivastav2205@gmail.com)
- 💼 LinkedIn: [in/sauravsrivastav2205](https://www.linkedin.com/in/sauravsrivastav2205)
- 🐙 GitHub: [https://github.com/SauravSrivastav](https://github.com/SauravSrivastav)

---
Happy Chatting! 🎉
