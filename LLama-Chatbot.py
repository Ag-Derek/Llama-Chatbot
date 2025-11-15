#!/usr/bin/env python
# coding: utf-8

import os
import requests
import streamlit as st

st.title("Simple Llama 3.2 Chatbot")

# Load HF API key
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    st.error("Please set HF_API_KEY as an environment variable.")
    st.stop()

# Use a model supported by HF Inference Provider
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# The NEW Hugging Face inference router endpoint
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def ask_model(prompt: str):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 128
        }
    }

    try:
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    except Exception as e:
        return {
            "error": str(e),
            "status_code": resp.status_code if 'resp' in locals() else None,
            "text": resp.text if 'resp' in locals() else None
        }

prompt = st.text_area("Your message", height=150)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Write a message first.")
    else:
        with st.spinner("Waiting for Llama 3.2..."):
            result = ask_model(prompt)

            # Error display
            if isinstance(result, dict) and result.get("error"):
                st.error(result)

            # Normal inference output
            else:
                try:
                    st.write(result[0]["generated_text"])
                except Exception:
                    st.write(result)
