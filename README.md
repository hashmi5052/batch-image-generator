# LoRA Image Generator App

This app lets you generate images using local LoRA and model checkpoints based on 1000 prompts from a `.csv` or `.txt` file.

## 🧰 Requirements
- Python 3.10
- Windows 10/11
- NVIDIA GPU (optional for faster results)

## 🚀 Setup Instructions

1. Clone the repo:
git clone https://github.com/YOUR_USERNAME/lora-image-generator.git
cd lora-image-generator

2. Install dependencies:

pip install -r requirements.txt


3. Run the app:
python app.py

4. Upload your `models/checkpoints/` and `models/lora/` manually into the folder.

## 📁 Folder Structure
- `models/checkpoints/` → Put your `.safetensors` base models here
- `models/lora/` → Put your LoRA files here
- `output/` → Images will be saved here
