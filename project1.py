import warnings
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")  # Hide warnings

# Page config
st.set_page_config(page_title="AI Text Generator", layout="centered")

st.title("💬 Text Generator")
st.write("Select a prompt and generate AI text")

# Model setup
model_id = "distilgpt2"

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
)

# 5 Predefined Prompts
prompts = [
    "In the future, education will",
    "Artificial intelligence will change",
    "Technology in 2030 will",
    "The role of teachers will",
    "Online learning will become",
]

# User selection dropdown
selected_prompt = st.selectbox("Choose a prompt:", prompts)

st.write("### Selected Prompt:")
st.info(selected_prompt)

# Generate button and logic
if st.button("Generate Text"):
    output = generator(
        selected_prompt,
        max_length=60,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tok.pad_token_id,
    )
    
    st.subheader("Generated Text:")
    st.success(output["generated_text"])
