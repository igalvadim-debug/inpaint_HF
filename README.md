---
title: Image Upscaler
emoji: 🖼️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🖼️ Image Upscaler

A modern web interface for upscaling images using advanced AI models. This application uses state-of-the-art upscaling algorithms to enhance image quality while maintaining sharpness and detail.

## Features

- 🔥 **Modern Architecture**: Built with Gradio 4.x and spandrel (ComfyUI standard)
- 🧠 **Multiple Models**: Supports RealESRGAN models including x4plus and anime_6B variants
- 💻 **Smart Device Detection**: Automatically detects and uses GPU (CUDA/MPS) or falls back to CPU
- 🧹 **Memory Management**: Includes tiling and garbage collection for handling large images
- 🌐 **Hugging Face Spaces Ready**: Optimized for deployment on CPU containers

## Supported Models

- `RealESRGAN_x4plus`: General purpose 4x upscaling
- `RealESRGAN_x4plus_anime_6B`: Optimized for anime-style images
- `RealESRGAN_x2plus`: General purpose 2x upscaling

## How to Use

1. Upload an image using the input area
2. Select an upscaling model from the dropdown
3. Click "Upscale Image" to process
4. View the enhanced result in the output area

## Technical Details

- Uses spandrel library (standard in ComfyUI) for model loading
- Falls back to realesrgan if spandrel is unavailable
- Implements tiling to handle large images without memory overflow
- Automatically downloads models on first use
- Optimized for CPU execution on Hugging Face Spaces

## Requirements

- Python 3.8+
- PyTorch with CUDA support (optional, falls back to CPU)
- See `requirements.txt` for complete dependency list

## Local Installation

```bash
pip install -r requirements.txt
python app.py
```

The application will automatically download required models on first use to the `models/` directory.