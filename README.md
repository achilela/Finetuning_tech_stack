## Valony Finetuning Labs 🚀
Welcome to Valony Finetuning Labs, an interactive Streamlit application for fine-tuning language models like SmolLM2 using various frameworks (Hugging Face, Unsloth, Axolotl). This app allows users to load datasets from Hugging Face or upload local files, configure training parameters, and push fine-tuned models to Hugging Face, Ollama, or GitHub. The interface uses the TW Cen MT font for a sleek, professional look.
Overview
Valony Finetuning Labs is designed to simplify the process of fine-tuning small language models (e.g., SmolLM2) for specific tasks. It provides a user-friendly, low-code environment with options for:
Selecting a model (SmolLM2 variants: 135M, 360M, 1.7B).
Choosing a fine-tuning framework (Hugging Face, Unsloth, Axolotl).
Loading datasets from Hugging Face or uploading local files (.txt, .json, .jsonl, .csv).
Configuring training parameters (e.g., max steps, learning rate).
Pushing the fine-tuned model to Hugging Face, Ollama, or GitHub.
The app features a highly interactive interface with real-time feedback, styled in TW Cen MT font throughout.
Features
Interactive Dashboard: Enter prompts, fine-tune models, and test responses in real-time.
Flexible Dataset Input: Support for Hugging Face datasets or local file uploads.
Multiple Frameworks: Choose between Hugging Face’s Transformers, Unsloth, or Axolotl for fine-tuning.
Model Push Options: Push fine-tuned models to Hugging Face, Ollama (placeholder), or GitHub.
Downloadable Output: Download the fine-tuned model as a .zip file.
Custom Styling: Uses TW Cen MT font for a consistent, professional look across the app.
Prerequisites
Python 3.8 or higher
Internet connection (for accessing Hugging Face datasets and Google Fonts)
Installation
Clone or Download the Repository
If you have the code locally, navigate to the directory containing valony_finetuning_labs.py.
Install Dependencies
Install the required Python packages using the provided requirements.txt:
bash
pip install -r requirements.txt
Verify Environment
Ensure you have Git installed (for GitHub push functionality) and your network allows access to huggingface.co and fonts.googleapis.com.
Usage
Run the App
Launch the Streamlit app from your terminal:
bash
streamlit run valony_finetuning_labs.py
Configure Settings  
Use the sidebar to select a model, framework, dataset source, and training parameters.
For Hugging Face datasets, enter the dataset name (e.g., amiguel/class) and optionally provide a Hugging Face token if required.
For local files, upload .txt, .json, .jsonl, or .csv files containing your dataset.
Finetune and Test  
Enter a test prompt in the main dashboard.
Click “Start Finetuning” to train the model.
View the response from the fine-tuned model and download or push it as needed.
Push Options  
Push the model to Hugging Face (requires repo name and token).
Push to Ollama (placeholder functionality, requires username/password).
Push to GitHub (requires token and repo name).
Dependencies
The project uses the following Python packages (see requirements.txt for exact versions):
streamlit: For the web interface
transformers: For model handling and fine-tuning
datasets: For loading Hugging Face datasets
torch: For PyTorch-based model training
pandas: For handling CSV files
huggingface_hub: For interacting with Hugging Face
requests: For SSL handling and network requests
Troubleshooting
SSLError for Hugging Face Datasets: If you encounter SSL errors when loading datasets, check the “Bypass SSL Verification” option in the sidebar (use cautiously in secure environments) or provide a Hugging Face token.
Font Issues (TW Cen MT): Ensure your network allows access to fonts.googleapis.com. If the font doesn’t load, the app falls back to Arial, Helvetica, or sans-serif. For offline use, download and host the TW Cen MT font locally (see code comments for instructions).
Duplicate Element ID Error: Ensure all Streamlit elements (e.g., st.text_input, st.button) have unique key arguments, as provided in the code.
Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss.
License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable).
Contact
For questions or support, contact Valony Labs at your-email@example.com (mailto:your-email@example.com) or visit our website at valonylabs.com.

