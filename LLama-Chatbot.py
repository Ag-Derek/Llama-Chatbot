#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import streamlit as st

st.title("Simple Llama 3.2 Chatbot")

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    st.error("Please set HF_API_KEY as an environment variable.")
    st.stop()

MODEL = "google/flan-t5-small"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def ask_model(prompt: str):
    payload = {"inputs": prompt}
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Request failed: {e}", "status_code": resp.status_code, "text": resp.text}
    return resp.json()

prompt = st.text_area("Your message", height=150)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Write a message first.")
    else:
        with st.spinner("Waiting for model..."):
            out = ask_model(prompt)
            if isinstance(out, dict) and out.get("error"):
                st.error(out)
            else:
                # expected HF inference returns a list with generated_text
                try:
                    st.write(out[0]["generated_text"])
                except Exception:
                    st.write(out)

