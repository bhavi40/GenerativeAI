# MiniLLM: Hugging Face Edition

A lightweight Python script for interacting with Large Language Models (LLMs) using the **Hugging Face Inference API**. This project allows you to chat with state-of-the-art open-source models like Llama 3.1 directly from your terminal without needing to host the models locally.

## 🚀 Features

* **Serverless Inference:** Run powerful models (Llama, Mistral, Qwen) using Hugging Face's hosted infrastructure.
* **Conversational Interface:** Uses the `chat_completion` method to maintain structured dialogue with System and User roles.
* **Adjustable Parameters:** Easily tune `temperature` for creativity and `max_tokens` for response length.

## 🛠️ Setup

### 1. Installation

Ensure you have the `huggingface_hub` library installed:

```bash
pip install huggingface_hub

```

### 2. Get Your API Token

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Go to **Settings > Access Tokens** and create a "Read" token.
3. Paste your token into the `client` initialization in `minillm.py`.

## 💻 Usage

Run the script to start an interactive chat session in your terminal:

```bash
python minillm.py

```

## 📖 Key LLM Concepts in this Code

* **Model ID:** We are using `meta-llama/Llama-3.1-8B-Instruct`, a highly capable model for chat-based tasks.
* **System Prompt:** Set in the `messages` list to define the AI's persona as a "helpful assistant".
* **Temperature (0.7):** Provides a balance between factual consistency and natural, varied language.
* **Max Tokens (200):** Limits the response length to keep the conversation concise and save on API usage.

