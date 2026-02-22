import os
import gc
import torch
import numpy as np
from PIL import Image
import requests
from urllib.parse import urlparse
import gradio as gr

try:
    # Using spandrel as requested (modern approach used in ComfyUI)
    from spandrel import ModelLoader, ImageModelDescriptor
    HAS_SPANDREL = True
except ImportError:
    HAS_SPANDREL = False
    print("Spandrel not available, trying realesrgan...")

# Fallback to realesrgan if spandrel is not available
try:
    from realesrgan import RealESRGANer
    HAS_REALESRGAN = True
except ImportError:
    HAS_REALESRGAN = False
    print("Real-ESRGAN not available")

# Define available models with their download links
AVAILABLE_MODELS = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4
    },
    "RealESRGAN_x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2
    }
}

def download_model(model_name, model_info):
    """Download model if it doesn't exist locally"""
    model_path = f"models/{model_name}.pth"
    
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        print(f"Downloading {model_name}...")
        
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()
        
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {model_name} to {model_path}")
    
    return model_path

def get_device():
    """Detect available device (GPU or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class Upscaler:
    def __init__(self):
        self.current_model = None
        self.model_descriptor = None
        self.model_upscaler = None  # For realesrgan fallback
        self.device = get_device()
        print(f"Using device: {self.device}")
    
    def load_model(self, model_name):
        """Load the selected model using either spandrel or realesrgan"""
        if self.current_model == model_name:
            return  # Already loaded
        
        model_info = AVAILABLE_MODELS[model_name]
        model_path = download_model(model_name, model_info)
        
        # Clear previous models
        self.model_descriptor = None
        self.model_upscaler = None
        
        # First try to load with spandrel
        if HAS_SPANDREL:
            try:
                model_loader = ModelLoader(self.device)
                self.model_descriptor = model_loader.load_from_file(model_path)
                
                # Move model to device
                self.model_descriptor.to(self.device)
                self.model_descriptor.eval()
                
                self.current_model = model_name
                print(f"Loaded model {model_name} with spandrel successfully")
                return
                
            except Exception as e:
                print(f"Error loading model with spandrel: {e}")
        
        # Fallback to realesrgan if spandrel fails or is not available
        if HAS_REALESRGAN:
            try:
                # Get scale factor from model info
                scale_factor = model_info["scale"]
                
                # Initialize Real-ESRGAN upscaler
                self.model_upscaler = RealESRGANer(
                    scale=scale_factor,
                    model_path=model_path,
                    dni_weight=None,
                    model_name=model_name,
                    half=False,  # Set to True for better performance on some GPUs
                    device=self.device
                )
                
                self.current_model = model_name
                print(f"Loaded model {model_name} with realesrgan successfully")
                return
                
            except Exception as e:
                print(f"Error loading model with realesrgan: {e}")
        
        # If both methods failed
        raise gr.Error(f"Failed to load model {model_name} with both spandrel and realesrgan: {str(e)}")
    
    def upscale_image(self, image, model_name, tile_size=1024):
        """Upscale image with tiling to manage memory"""
        if image is None:
            raise gr.Error("No image provided")
        
        # Load the selected model
        self.load_model(model_name)
        model_info = AVAILABLE_MODELS[model_name]
        scale_factor = model_info["scale"]
        
        # Use the appropriate upscaling method based on which library was loaded
        if self.model_descriptor is not None:  # Using spandrel
            return self._upscale_with_spandrel(image, scale_factor, tile_size)
        elif self.model_upscaler is not None:  # Using realesrgan
            return self._upscale_with_realesrgan(image, scale_factor)
        else:
            raise gr.Error("No valid model loaded for upscaling")
    
    def _upscale_with_spandrel(self, image, scale_factor, tile_size=1024):
        """Upscale using spandrel model"""
        # Convert PIL image to numpy array
        img_np = np.array(image)
        
        # Handle different image formats
        if len(img_np.shape) == 2:  # Grayscale
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = img_np[:, :, :3]  # Convert to RGB
        
        # Normalize to [0, 1] range
        img_np = img_np.astype(np.float32) / 255.0
        
        # Process with tiling to avoid memory issues
        h, w, c = img_np.shape
        tile_size = min(tile_size, max(h, w))  # Adjust tile size if needed
        
        # Add batch and channel dimensions for PyTorch (B, C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Convert to tensor and move to device
        img_tensor = img_tensor.to(self.device, dtype=torch.float32)
        
        # Perform upscaling with tiling if needed
        upscaled_tensor = self._process_with_tiling(img_tensor, scale_factor, tile_size)
        
        # Move result back to CPU and convert to numpy
        upscaled_tensor = upscaled_tensor.cpu().clamp_(0, 1)
        upscaled_img = upscaled_tensor.squeeze(0).permute(1, 2, 0).numpy()
        
        # Convert back to uint8
        upscaled_img = (upscaled_img * 255.0).round().astype(np.uint8)
        
        # Convert to PIL Image
        result_image = Image.fromarray(upscaled_img)
        
        # Clean up memory
        del img_tensor, upscaled_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return result_image
    
    def _upscale_with_realesrgan(self, image, scale_factor):
        """Upscale using realesrgan model"""
        # Convert PIL image to numpy array (RGB)
        img_np = np.array(image)
        
        # Handle different image formats
        if len(img_np.shape) == 2:  # Grayscale
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 4:  # RGBA
            img_np = img_np[:, :, :3]  # Convert to RGB
        
        # Perform upscaling with realesrgan
        try:
            # Upscale the image
            upscaled_img, _ = self.model_upscaler.enhance(img_np)
            
            # Convert to PIL Image
            result_image = Image.fromarray(upscaled_img)
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result_image
        except Exception as e:
            raise gr.Error(f"Error during upscaling with Real-ESRGAN: {str(e)}")
    
    def _process_with_tiling(self, img_tensor, scale_factor, tile_size):
        """Process image with tiling to manage memory"""
        b, c, h, w = img_tensor.shape
        
        # If image is small enough, process it directly
        if h <= tile_size and w <= tile_size:
            return self._upscale_single_tile(img_tensor, scale_factor)
        
        # Calculate number of tiles needed
        tile_h = min(tile_size, h)
        tile_w = min(tile_size, w)
        
        # Calculate overlap to reduce artifacts
        overlap_h = tile_h // 8
        overlap_w = tile_w // 8
        
        # Create output tensor
        output_h, output_w = h * scale_factor, w * scale_factor
        output_tensor = torch.zeros((b, c, output_h, output_w), 
                                   dtype=img_tensor.dtype, 
                                   device=img_tensor.device)
        
        # Process each tile
        for i in range(0, h, tile_h - overlap_h * 2):
            for j in range(0, w, tile_w - overlap_w * 2):
                # Define tile boundaries
                start_h, end_h = i, min(i + tile_h, h)
                start_w, end_w = j, min(j + tile_w, w)
                
                # Add overlap
                start_h = max(0, start_h - overlap_h)
                end_h = min(h, end_h + overlap_h)
                start_w = max(0, start_w - overlap_w)
                end_w = min(w, end_w + overlap_w)
                
                # Extract tile
                tile = img_tensor[:, :, start_h:end_h, start_w:end_w]
                
                # Upscale tile
                upscaled_tile = self._upscale_single_tile(tile, scale_factor)
                
                # Calculate position in output
                out_start_h, out_end_h = start_h * scale_factor, end_h * scale_factor
                out_start_w, out_end_w = start_w * scale_factor, end_w * scale_factor
                
                # Place tile in output
                output_tensor[:, :, out_start_h:out_end_h, out_start_w:out_end_w] = upscaled_tile
        
        return output_tensor
    
    def _upscale_single_tile(self, tile_tensor, scale_factor):
        """Upscale a single tile"""
        with torch.no_grad():
            if hasattr(self.model_descriptor, 'model'):
                upscaled = self.model_descriptor.model(tile_tensor)
            else:
                # Fallback if model_descriptor is just the model
                upscaled = self.model_descriptor(tile_tensor)
        return upscaled

# Global upscaler instance
upscaler = Upscaler()

def process_image(image, model_name):
    """Main processing function called by Gradio"""
    if image is None:
        raise gr.Error("Please upload an image first")
    
    try:
        result = upscaler.upscale_image(image, model_name)
        return result
    except Exception as e:
        raise gr.Error(f"Error processing image: {str(e)}")

with gr.Blocks(title="Image Upscaler", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ Image Upscaler")
    gr.Markdown("Upload an image and select a model to upscale it.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            model_selector = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value="RealESRGAN_x4plus",
                label="Upscaling Model"
            )
            upscale_btn = gr.Button("Upscale Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Upscaled Image", interactive=False)
    
    upscale_btn.click(
        fn=process_image,
        inputs=[input_image, model_selector],
        outputs=output_image
    )
    
    gr.Markdown("### How to use:")
    gr.Markdown("- Upload an image using the input box")
    gr.Markdown("- Select an upscaling model from the dropdown")
    gr.Markdown("- Click 'Upscale Image' to process")
    gr.Markdown("- The upscaled image will appear in the output box")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))