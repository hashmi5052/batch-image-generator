import os
import csv
import pandas as pd
import gradio as gr
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from compel import Compel

# Define folders
CHECKPOINT_FOLDER = "models/checkpoints"
LORA_FOLDER = "models/lora"
OUTPUT_FOLDER = "output"
STOP_FLAG = False  # Global stop flag

# Prompt enhancements
POSITIVE_PHRASES = "masterpiece, highly detailed, cinematic lighting, DSLR, bokeh, shallow depth of field, studio lighting, photo, realistic"
NEGATIVE_PROMPT = "blurry, low quality, bad anatomy, extra limbs, worst quality, low resolution, distorted, grainy, watermark"

# Utility: List safetensors in folder
def list_safetensors(path):
    return [f for f in os.listdir(path) if f.endswith(".safetensors")]

# Read prompts and image numbers from the uploaded file
def read_prompt_file(file_path):
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        df = pd.read_csv(file_path, names=["prompt"])
        df["image number"] = range(1, len(df) + 1)
        return df
    else:
        raise ValueError("Unsupported file type.")

    df.columns = df.columns.str.strip().str.lower()
    required_columns = ["prompt", "image number"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}. Found columns: {df.columns.tolist()}")

    df = df.dropna(subset=["prompt"])
    df["prompt"] = df["prompt"].astype(str)
    df["image number"] = df["image number"].astype(int)
    return df

# Auto-detect starting number
def get_starting_number(file):
    try:
        df = read_prompt_file(file.name)
        return int(df["image number"].iloc[0])
    except Exception as e:
        print(f"Error detecting starting number: {e}")
        return 0

# Create unique output folder
def create_output_subfolder():
    existing = [d for d in os.listdir(OUTPUT_FOLDER) if d.startswith("run_")]
    existing_nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_num = max(existing_nums, default=0) + 1
    subfolder = os.path.join(OUTPUT_FOLDER, f"run_{next_num:03}")
    os.makedirs(subfolder, exist_ok=True)
    return subfolder

# Load model + LoRA
def load_pipeline(model_path, lora_path):
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    if lora_path:
        state_dict = load_file(lora_path)
        pipe.load_lora_weights(state_dict)
    pipe.enable_model_cpu_offload()
    return pipe

# Stop button function
def stop_generation():
    global STOP_FLAG
    STOP_FLAG = True
    return "⛔ Generation stopped by user."

# Reset stop flag
def reset_flag():
    global STOP_FLAG
    STOP_FLAG = False

# Main generation logic
def generate_images_stream(prompt_file, model_name, lora_name, image_format,
                           sampling_steps, guidance_scale, width, height,
                           resume_choice, upscale_model):
    global STOP_FLAG
    reset_flag()
    df = read_prompt_file(prompt_file.name)
    output_subfolder = create_output_subfolder()

    # Load pipeline
    model_path = os.path.join(CHECKPOINT_FOLDER, model_name)
    lora_path = os.path.join(LORA_FOLDER, lora_name) if lora_name else None
    pipe = load_pipeline(model_path, lora_path)
    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    resume_file = "last_image_number.txt"
    last_number = 0
    if resume_choice == "Resume from last saved":
        try:
            with open(resume_file, "r") as f:
                last_number = int(f.read().strip())
        except:
            last_number = 0

    df = df[df["image number"] > last_number]
    df = df.sort_values("image number")

    for idx, row in enumerate(df.iterrows()):
        if idx >= 100 or STOP_FLAG:
            break

        row = row[1]
        prompt = f"{POSITIVE_PHRASES}, {row['prompt']}"
        number = row["image number"]
        prompt_embeds = compel_proc(prompt)
        neg_embeds = compel_proc(NEGATIVE_PROMPT)

        image = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            num_inference_steps=int(sampling_steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            added_cond_kwargs={}  # Ensures proper structure
        ).images[0]

        # Placeholder for future upscaling
        if upscale_model != "None":
            # Apply GFPGAN / CodeFormer / Real-ESRGAN here (backend processing)
            pass

        filename = f"{number}.{image_format}"
        image_path = os.path.join(output_subfolder, filename)
        image.save(image_path)

        with open(resume_file, "w") as f:
            f.write(str(number))

        yield f"✅ Image {number} saved."

    yield f"🎉 Done. {idx + 1} image(s) saved in: {output_subfolder}"

# Gradio UI
def gradio_ui():
    with gr.Blocks(title="LoRA Batch Generator") as demo:
        gr.Markdown("## 🖼️ Local Batch Image Generator with Numbered Outputs")

        with gr.Row():
            prompt_file = gr.File(label="📄 Upload Prompt File (.csv, .xlsx, .txt)", file_types=[".csv", ".xlsx", ".txt"])
            model_dropdown = gr.Dropdown(label="🧠 Select Checkpoint", choices=list_safetensors(CHECKPOINT_FOLDER))
            lora_dropdown = gr.Dropdown(label="🎨 Select LoRA (Optional)", choices=[""] + list_safetensors(LORA_FOLDER))

        with gr.Row():
            start_index = gr.Number(label="🔢 First Image Number (Auto-detected)", value=0, interactive=False)
            image_format = gr.Dropdown(label="🖼️ Image Format", choices=["webp", "png"], value="webp")

        with gr.Row():
            sampling_steps = gr.Number(label="⏱️ Sampling Steps", value=30)
            guidance_scale = gr.Number(label="🎯 Guidance Scale", value=7.5)

        with gr.Row():
            width = gr.Number(label="📏 Width", value=512)
            height = gr.Number(label="📐 Height", value=512)

        resume_choice = gr.Dropdown(
            label="🔁 Resume Option",
            choices=["Start from first", "Resume from last saved"],
            value="Start from first"
        )

        upscale_model = gr.Dropdown(
            label="🧼 Upscale Model (Optional)",
            choices=["None", "GFPGAN", "Real-ESRGAN", "CodeFormer"],
            value="None"
        )

        with gr.Row():
            run_button = gr.Button("🚀 Start Generating")
            stop_button = gr.Button("🛑 Stop Generation")

        output_text = gr.Textbox(label="📢 Status Log", lines=15)

        prompt_file.change(fn=get_starting_number, inputs=[prompt_file], outputs=start_index)

        run_button.click(
            fn=generate_images_stream,
            inputs=[
                prompt_file, model_dropdown, lora_dropdown, image_format,
                sampling_steps, guidance_scale, width, height,
                resume_choice, upscale_model
            ],
            outputs=output_text
        )

        stop_button.click(fn=stop_generation, outputs=output_text)

    return demo

if __name__ == "__main__":
    gradio_ui().launch(share=True)
