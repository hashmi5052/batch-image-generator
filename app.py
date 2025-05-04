import os
import csv
import gradio as gr
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from compel import Compel

# Define paths
CHECKPOINT_FOLDER = "models/checkpoints"
LORA_FOLDER = "models/lora"
OUTPUT_FOLDER = "output"

# Util to list safetensors files from directory
def list_safetensors(path):
    return [f for f in os.listdir(path) if f.endswith(".safetensors")]

# Load prompt list
def load_prompts(file_path):
    prompts = []
    if file_path.endswith(".csv"):
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    prompts.append(row[0])
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as txtfile:
            prompts = txtfile.read().splitlines()
    return prompts

# Load model and apply LoRA
def load_pipeline(model_path, lora_path):
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Apply LoRA manually
    if lora_path:
        state_dict = load_file(lora_path)
        pipe.load_lora_weights(state_dict)

    pipe.enable_model_cpu_offload()  # optional for large models
    return pipe

# Main batch generation logic
def generate_images(prompt_file, model_name, lora_name, start_index, image_format):
    prompts = load_prompts(prompt_file.name)
    start_index = int(start_index)
    
    output_subfolder = os.path.join(OUTPUT_FOLDER, str(start_index))
    os.makedirs(output_subfolder, exist_ok=True)

    model_path = os.path.join(CHECKPOINT_FOLDER, model_name)
    lora_path = os.path.join(LORA_FOLDER, lora_name) if lora_name else None

    pipe = load_pipeline(model_path, lora_path)
    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        prompt_embeds = compel_proc(prompt)
        image = pipe(prompt_embeds=prompt_embeds).images[0]
        filename = f"{start_index + i}.{image_format}"
        image_path = os.path.join(output_subfolder, filename)
        image.save(image_path)

    return f"✅ Generated {len(prompts)} images in folder: {output_subfolder}"

# ---------- Gradio UI ----------
def gradio_ui():
    with gr.Blocks(title="LoRA Batch Generator") as demo:
        gr.Markdown("## 🖼️ Local Batch Image Generator (with LoRA + Checkpoint)")

        with gr.Row():
            prompt_file = gr.File(label="📄 Upload .csv or .txt Prompt File", file_types=[".csv", ".txt"])
            model_dropdown = gr.Dropdown(label="🧠 Select Checkpoint", choices=list_safetensors(CHECKPOINT_FOLDER))
            lora_dropdown = gr.Dropdown(label="🎨 Select LoRA (Optional)", choices=[""] + list_safetensors(LORA_FOLDER))
        
        with gr.Row():
            start_index = gr.Number(label="🔢 Starting Filename Number", value=891)
            image_format = gr.Dropdown(label="📁 Image Format", choices=["webp", "png"], value="webp")

        run_button = gr.Button("🚀 Start Generating")
        output_text = gr.Textbox(label="📢 Status", lines=3)

        run_button.click(fn=generate_images,
                         inputs=[prompt_file, model_dropdown, lora_dropdown, start_index, image_format],
                         outputs=output_text)

    return demo

if __name__ == "__main__":
    gradio_ui().launch()
