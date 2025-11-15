import os
import requests
import streamlit as st

st.title("Simple Chatbot (HF Inference Provider)")

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    st.error("Please set HF_API_KEY.")
    st.stop()

MODEL = "microsoft/Phi-3-mini-4k-instruct"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def ask_model(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100}
    }
    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "status_code": r.status_code, "text": r.text}

prompt = st.text_area("Your message")

if st.button("Generate"):
    with st.spinner("Thinking..."):
        out = ask_model(prompt)
        if "error" in out:
            st.error(out)
        else:
            st.write(out[0]["generated_text"])
