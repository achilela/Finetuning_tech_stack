import streamlit as st
from streamlit.components.v1 import html
import os
import subprocess
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset, DownloadConfig
import pandas as pd
import json
import ssl
import requests

# Custom CSS for TW Cen MT font with fallback (including sidebar)
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Twentieth+Century:wght@400;700&display=swap');
body, h1, h2, h3, p, label, .stTextInput, .stSelectbox, .stRadio, .stFileUploader, .stButton, .stDownloadButton, .sidebar .sidebar-content {
    font-family: 'Twentieth Century', Arial, Helvetica, sans-serif !important;
}
h1 {
    text-align: center;
}
</style>
"""

# Inject custom CSS
html(custom_css)

# Title with rocket emojis
st.markdown("🚀 **Valony Finetuning Labs** 🚀", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("Finetuning Configuration")

# Model selection
model_options = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
]
selected_model = st.sidebar.selectbox("Select Model to Finetune", model_options)

# Framework selection
framework_options = ["Tutorial (Hugging Face)", "Unsloth", "Axolotl"]
selected_framework = st.sidebar.radio("Select Finetuning Framework", framework_options)

# Dataset source selection
data_source = st.sidebar.radio("Dataset Source", ["Hugging Face Repo", "Upload File"])

# Dataset input based on source
dataset = None
if data_source == "Hugging Face Repo":
    dataset_name = st.sidebar.text_input("Enter Hugging Face Dataset Name", "amiguel/class", key="dataset_name_input")
    hf_token = st.sidebar.text_input("Hugging Face Token (optional)", "", type="password", key="hf_token_input")
    bypass_ssl = st.sidebar.checkbox("Bypass SSL Verification (for SSLError)", False, key="bypass_ssl_checkbox")

    if dataset_name:
        try:
            # Configure download settings
            download_config = DownloadConfig(use_auth_token=hf_token if hf_token else None)
            if bypass_ssl:
                # Disable SSL verification (use with caution in secure environments)
                requests.packages.urllib3.disable_warnings()
                download_config.session = requests.Session()
                download_config.session.verify = False

            dataset = load_dataset(dataset_name, download_config=download_config)
            st.sidebar.success(f"Loaded dataset: {dataset_name}")
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Dataset File (.txt, .json, .jsonl, .csv)", type=["txt", "json", "jsonl", "csv"], key="file_uploader")
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "txt":
            dataset = uploaded_file.read().decode("utf-8").splitlines()
        elif file_extension == "json":
            dataset = json.load(uploaded_file)
        elif file_extension == "jsonl":
            dataset = [json.loads(line) for line in uploaded_file.read().decode("utf-8").splitlines()]
        elif file_extension == "csv":
            dataset = pd.read_csv(uploaded_file).to_dict(orient="records")
        st.sidebar.success(f"Uploaded file: {uploaded_file.name}")

# Training parameters
max_steps = st.sidebar.slider("Max Training Steps", 10, 1000, 100, key="max_steps_slider")
learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-6, max_value=1e-3, value=2e-5, format="%.6f", key="learning_rate_input")

# Push options
st.sidebar.header("Model Push Options")
hf_repo = st.sidebar.text_input("Hugging Face Repo (e.g., username/repo)", "", key="hf_repo_input")
hf_token_push = st.sidebar.text_input("Hugging Face Token (optional)", "", type="password", key="hf_token_push_input")

ollama_push = st.sidebar.checkbox("Push to Ollama", key="ollama_push_checkbox")
if ollama_push:
    ollama_username = st.sidebar.text_input("Ollama Username", "", key="ollama_username_input")
    ollama_password = st.sidebar.text_input("Ollama Password", "", type="password", key="ollama_password_input")

github_push = st.sidebar.checkbox("Push to GitHub", key="github_push_checkbox")
if github_push:
    github_token = st.sidebar.text_input("GitHub Token", "", type="password", key="github_token_input")
    github_repo = st.sidebar.text_input("GitHub Repo (e.g., username/repo)", "", key="github_repo_input")

# Main content
st.subheader("Finetuning Dashboard")

# Prompt interface
prompt = st.text_area("Enter a Test Prompt", "Explain AGI?", height=100, key="prompt_input")

# Finetuning logic
def finetune_model(model_name, dataset, framework, max_steps, lr):
    save_dir = "./finetuned_model"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    if framework == "Tutorial (Hugging Face)":
        st.write("Finetuning with Hugging Face Transformers...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if isinstance(dataset, dict) and "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset

        def tokenize_function(examples):
            return tokenizer(examples["text"] if "text" in examples else examples, padding="max_length", truncation=True)

        tokenized_dataset = train_data.map(tokenize_function, batched=True)
        
        from transformers import Trainer, TrainingArguments
        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            max_steps=max_steps,
            learning_rate=lr,
            logging_steps=10,
            save_steps=50,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    elif framework == "Unsloth":
        st.write("Finetuning with Unsloth (placeholder)...")
        # Add Unsloth-specific code here if available

    elif framework == "Axolotl":
        st.write("Finetuning with Axolotl (placeholder)...")
        # Add Axolotl-specific code here if available

    return save_dir

# Push logic
def push_to_huggingface(save_dir, repo_name, token):
    if not repo_name or not token:
        st.error("Please provide Hugging Face repo and token!")
        return
    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_name, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_name, repo_type="model")
    st.success(f"Model pushed to Hugging Face: {repo_name}")

def push_to_ollama(save_dir, username, password):
    if not username or not password:
        st.error("Please provide Ollama credentials!")
        return
    st.write("Pushing to Ollama (placeholder)...")
    st.success("Model pushed to Ollama (placeholder implementation)")

def push_to_github(save_dir, repo_name, token):
    if not repo_name or not token:
        st.error("Please provide GitHub repo and token!")
        return
    os.system(f"cd {save_dir} && git init && git add . && git commit -m 'Finetuned model' && git remote add origin https://{token}@github.com/{repo_name}.git && git push -u origin main --force")
    st.success(f"Model pushed to GitHub: {repo_name}")

# Finetune button
if st.button("Start Finetuning", key="finetune_button"):
    if dataset is None:
        st.error("Please provide a dataset!")
    else:
        with st.spinner("Finetuning in progress..."):
            save_dir = finetune_model(selected_model, dataset, selected_framework, max_steps, learning_rate)
            st.success(f"Model finetuned and saved to {save_dir}")

        # Test the finetuned model
        model = AutoModelForCausalLM.from_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        response = pipe(prompt, max_new_tokens=200)[0]["generated_text"]
        st.subheader("Response from Finetuned Model")
        st.write(response)

        # Push to selected platforms
        if hf_repo and hf_token_push:
            with st.spinner("Pushing to Hugging Face..."):
                push_to_huggingface(save_dir, hf_repo, hf_token_push)
        if ollama_push:
            with st.spinner("Pushing to Ollama..."):
                push_to_ollama(save_dir, ollama_username, ollama_password)
        if github_push:
            with st.spinner("Pushing to GitHub..."):
                push_to_github(save_dir, github_repo, github_token)

# Download finetuned model
if os.path.exists("./finetuned_model"):
    with open("./finetuned_model.zip", "wb") as f:
        shutil.make_archive("./finetuned_model", "zip", "./finetuned_model")
        f.write(open("./finetuned_model.zip", "rb").read())
    with open("./finetuned_model.zip", "rb") as f:
        st.download_button("Download Finetuned Model", f, file_name="finetuned_model.zip", key="download_button")

# Footer
st.markdown("---")
st.write("Built with ❤️ by Finetuning Labs | Powered by ValonyLabs")
