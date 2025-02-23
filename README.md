# FaceSpark: Zero-Shot Facial Expression Recognition

## Project Overview
FaceSpark is a lightweight, zero-shot facial expression recognition (FER) system that combines large language models (LLMs) and vision-language models (VLMs) like CLIP. It identifies emotions (happy, sad, angry, surprised, fearful, disgusted, neutral) from webcam or images without labeled data, optimized for 6GB GPUs. The project features a sleek Gradio UI for real-time interaction, ideal for showcasing NLP, CV, and model optimization skills in interviews.

## Features
- **Zero-Shot FER**: Uses CLIP and LLM-generated descriptions for emotion recognition without training on labeled data.
- **Real-Time Processing**: Captures and analyzes facial expressions from webcam or uploaded images.
- **Lightweight Design**: Runs on 6GB GPUs, avoiding heavy dependencies like 8-bit quantization.
- **Interactive Interface**: Stylish Gradio UI with modern layout, video streaming, and expression display.
- **Minimal Logging**: Training and inference use sparse logs, no progress bar clutter.

## LLM-Related Knowledge Points
- **Zero-Shot Learning**: Leverages pre-trained CLIP for visual-text alignment, enhanced by LLMs for task-specific knowledge.
- **Vision-Language Models (VLMs)**: Uses CLIP to bridge image and text for FER, inspired by Exp-CLIP and DFER-CLIP.
- **LLM Knowledge Transfer**: Generates expression descriptions with LLMs (simulated via hardcoded data).
- **Computer Vision**: Employs OpenCV for real-time webcam capture and processing.
- **PyTorch Optimization**: Uses native PyTorch for GPU/CPU compatibility, minimizing resource use.

## Environment Setup and Deployment

### Prerequisites
- **Hardware**: GPU with at least 6GB VRAM (e.g., NVIDIA GTX 1660) or CPU.
- **Operating System**: Windows (tested on Windows 10/11).
- **Python**: 3.9 or higher.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AzuCai/FaceSpark.git
   cd FaceSpark
   ```
2. **Create a Conda Environment (optional but recommended)**:
  ```bash
   conda create -n FaceSpark python=3.10
   conda activate FaceSpark
   ```
3. **Install Dependencies**:
   ```bash
   pip install transformers torch opencv-python gradio
   ```
### Running the Project
1. **Launch and Test**:
  ```bash
  python main.py
  ```
2. **Interact**:
  Upload images or start the webcam to detect expressions like “happy” or “sad.”

## Acknowledgments
Built with Hugging Face Transformers, OpenCV, and Gradio.

Inspired by Exp-CLIP and DFER-CLIP for LLM-enhanced FER.

Enjoy exploring FaceSpark!
