import os
import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
from PIL import Image

# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(model_name).to(device)

# Define facial expression descriptions
def get_expression_descriptions():
    """
    Generate textual descriptions for facial expressions
    :return: List of expressions and their descriptions
    """
    expressions = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral"]
    descriptions = {
        "happy": "A happy expression shows a smile, raised cheeks, and crinkled eyes.",
        "sad": "A sad expression features downturned mouth, furrowed brows, and teary eyes.",
        "angry": "An angry expression shows furrowed brows, narrowed eyes, and a clenched jaw.",
        "surprised": "A surprised expression has wide eyes, raised eyebrows, and an open mouth.",
        "fearful": "A fearful expression shows wide eyes, raised eyebrows, and a tensed mouth.",
        "disgusted": "A disgusted expression features a wrinkled nose, raised upper lip, and squinted eyes.",
        "neutral": "A neutral expression shows relaxed facial muscles, no strong emotion."
    }
    return expressions, descriptions

expressions, description_dict = get_expression_descriptions()

# Process an image using CLIP
def process_image(image):
    """
    Process an input image to extract visual features using CLIP
    :param image: PIL Image or NumPy array
    :return: Image features tensor
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features

# Zero-shot facial expression recognition using CLIP
def recognize_expression(image):
    """
    Recognize facial expression in an image using zero-shot learning
    :param image: PIL Image or NumPy array
    :return: Predicted expression
    """
    image_features = process_image(image)
    text_inputs = processor(text=list(description_dict.values()), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).softmax(dim=-1)

    max_idx = similarity.argmax().item()
    return expressions[max_idx]

# Webcam stream handler for real-time FER
def update_webcam(state=None):
    """
    Update webcam feed and expression in real-time using Gradio state
    :param state: Current state (None for initialization)
    :yield: Tuple of (frame as NumPy array, expression as str)
    """
    if state is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            yield None, "Error: Webcam not available"
            return
        state = {"cap": cap}

    cap = state["cap"]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                yield None, "Error: Failed to capture frame"
                break

            expression = recognize_expression(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame_rgb, f"Expression: {expression}"

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        yield None, f"Error: {e}"

    finally:
        cap.release()

# Define Gradio Interface with improved layout
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        color: #7f8c8d;
        font-size: 16px;
        text-align: center;
        margin-bottom: 30px;
    }
    .gr-button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .gr-button:hover {
        background-color: #2980b9;
    }
    .gr-image, .gr-video {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .gr-textbox {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #ddd;
    }
""") as interface:
    state = gr.State(value=None)

    with gr.Column(elem_id="main-column"):
        gr.Markdown(
            """
            <div class="title">FaceSpark: Zero-Shot Facial Expression Recognition</div>
            <div class="description">Upload an image or use the webcam for real-time zero-shot facial expression recognition using LLMs and CLIP.</div>
            """,
            elem_id="header"
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=200):
                webcam_button = gr.Button("Start Webcam", elem_classes="gr-button")
            with gr.Column(scale=2):
                webcam_image = gr.Image(label="Webcam Feed", elem_classes="gr-image")
            with gr.Column(scale=1, min_width=200):
                text_output = gr.Textbox(label="Expression", value="Start webcam to detect expression", elem_classes="gr-textbox")

        # Start webcam stream
        webcam_button.click(
            fn=update_webcam,
            inputs=[state],
            outputs=[webcam_image, text_output]
        )

# Launch Gradio interface
interface.launch(inbrowser=True, share=False)