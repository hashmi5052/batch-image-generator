# 🖼️ Batch Image Generator with LoRA & Model Selection

A local Python+Gradio app to generate bulk images from prompts using custom Stable Diffusion models and LoRA weights — with full offline support.

---

## 🚀 Features

- Upload a `.csv` or `.txt` file with up to 1000 prompts.
- Select models (checkpoints) and LoRAs from dropdown menus.
- Choose starting number for filenames (e.g., `891.webp`, `892.webp`, ...).
- Auto-creates folders under `output/` named after the starting number.
- Progress tracking inside the Gradio interface.
- Saves all images as `.webp` or `.png`.

---

## 🖥️ System Requirements

- Windows 10/11 (fully tested)
- Python 3.10
- GPU (Recommended: NVIDIA RTX 4080 or similar)
- All models and LoRA files stored **locally**

---


---

## 📦 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/hashmi5052/batch-image-generator.git
   cd batch-image-generator



2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install Dependencies

pip install -r requirements.txt


## 🧪 Usage Instructions

## Run the application

   python app.py



