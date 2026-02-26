#!/usr/bin/env python3
"""
CampusGen AI – Pipeline Manager
Centralized lazy-loading of all generation pipelines.
Shares base model + LoRA across text2img, img2img, inpainting.
Manages VRAM via CPU offloading for 16GB GPUs / HF ZeroGPU.
"""

import os
import gc
import logging
from typing import Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ─── SM120 (Blackwell) CUDA optimizations ───────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_USERNAME")
LORA_REPO = f"{HF_USERNAME}/campus-ai-poster-lora"
LORA_FILENAME = "campus_ai_poster_lora.safetensors"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"

# IP-Adapter for Flux
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"  # Flux-compatible adapter
IMAGE_ENCODER_REPO = "openai/clip-vit-large-patch14"

# Real-ESRGAN upscaler
ESRGAN_MODEL_NAME = "RealESRGAN_x4plus"


def flush_vram():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PipelineManager:
    """
    Manages all generation pipelines with shared base model.
    Only ONE pipeline mode is active at a time to fit in 16GB VRAM.
    """

    def __init__(self):
        self._text2img = None
        self._img2img = None
        self._inpaint = None
        self._ip_adapter_loaded = False
        self._upscaler = None
        self._active_mode: Optional[str] = None
        self._lora_loaded = False

    # ── Text-to-Image ────────────────────────────────────────────────────

    def get_text2img(self):
        """Load or return text-to-image pipeline."""
        if self._active_mode == "text2img" and self._text2img is not None:
            return self._text2img

        self._unload_all()

        from diffusers import FluxPipeline

        logger.info("Loading Flux.1-dev text-to-image pipeline...")
        self._text2img = FluxPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
        )
        self._text2img.enable_model_cpu_offload()
        self._load_lora(self._text2img)

        # SM120: compile transformer for faster inference
        try:
            self._text2img.transformer = torch.compile(
                self._text2img.transformer, mode="max-autotune"
            )
        except Exception:
            pass

        self._active_mode = "text2img"
        logger.info("Text-to-image pipeline ready.")
        return self._text2img

    # ── Image-to-Image ───────────────────────────────────────────────────

    def get_img2img(self):
        """Load or return img2img pipeline."""
        if self._active_mode == "img2img" and self._img2img is not None:
            return self._img2img

        self._unload_all()

        from diffusers import FluxImg2ImgPipeline

        logger.info("Loading Flux.1-dev img2img pipeline...")
        self._img2img = FluxImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
        )
        self._img2img.enable_model_cpu_offload()
        self._load_lora(self._img2img)

        try:
            self._img2img.transformer = torch.compile(
                self._img2img.transformer, mode="max-autotune"
            )
        except Exception:
            pass

        self._active_mode = "img2img"
        logger.info("Img2img pipeline ready.")
        return self._img2img

    # ── Inpainting ───────────────────────────────────────────────────────

    def get_inpaint(self):
        """Load or return inpainting pipeline."""
        if self._active_mode == "inpaint" and self._inpaint is not None:
            return self._inpaint

        self._unload_all()

        from diffusers import FluxInpaintPipeline

        logger.info("Loading Flux.1-dev inpainting pipeline...")
        self._inpaint = FluxInpaintPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
        )
        self._inpaint.enable_model_cpu_offload()
        self._load_lora(self._inpaint)

        try:
            self._inpaint.transformer = torch.compile(
                self._inpaint.transformer, mode="max-autotune"
            )
        except Exception:
            pass

        self._active_mode = "inpaint"
        logger.info("Inpainting pipeline ready.")
        return self._inpaint

    # ── IP-Adapter (style from reference image) ──────────────────────────

    def load_ip_adapter(self, pipe):
        """
        Attach IP-Adapter to the current pipeline for reference-image input.
        Uses CLIP image encoder to extract style features.
        """
        if self._ip_adapter_loaded:
            return pipe

        try:
            logger.info("Loading IP-Adapter for reference image support...")
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
            )
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load IP-Adapter: {e}")
            logger.warning("Reference image feature will be disabled.")

        return pipe

    def set_ip_adapter_scale(self, pipe, scale: float = 0.6):
        """Set the influence strength of the reference image."""
        if self._ip_adapter_loaded:
            pipe.set_ip_adapter_scale(scale)

    # ── Real-ESRGAN Upscaler ─────────────────────────────────────────────

    def get_upscaler(self):
        """Load and return the Real-ESRGAN upscaler model."""
        if self._upscaler is not None:
            return self._upscaler

        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            logger.info("Loading Real-ESRGAN x4 upscaler...")

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )

            self._upscaler = RealESRGANer(
                scale=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                model=model,
                tile=512,  # Tile size for memory-efficient upscaling
                tile_pad=10,
                pre_pad=0,
                half=True,  # FP16 for speed
            )
            logger.info("Real-ESRGAN upscaler ready.")

        except ImportError:
            logger.warning(
                "Real-ESRGAN not installed. Using Pillow LANCZOS fallback."
            )
            self._upscaler = "pillow_fallback"

        except Exception as e:
            logger.warning(f"Could not load Real-ESRGAN: {e}. Using fallback.")
            self._upscaler = "pillow_fallback"

        return self._upscaler

    def upscale_image(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """
        Upscale an image using Real-ESRGAN (or Pillow fallback).
        Input: PIL Image
        Output: PIL Image (upscaled)
        """
        upscaler = self.get_upscaler()

        if upscaler == "pillow_fallback":
            # Simple Pillow resize as fallback
            new_size = (image.width * scale, image.height * scale)
            return image.resize(new_size, Image.LANCZOS)

        # Real-ESRGAN
        img_np = np.array(image)
        # Real-ESRGAN expects BGR
        import cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        output, _ = upscaler.enhance(img_bgr, outscale=scale)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        return Image.fromarray(output_rgb)

    # ── LoRA Loading ─────────────────────────────────────────────────────

    def _load_lora(self, pipe):
        """Load LoRA weights onto a pipeline."""
        logger.info(f"Loading LoRA weights from {LORA_REPO}...")
        try:
            pipe.load_lora_weights(
                LORA_REPO,
                weight_name=LORA_FILENAME,
            )
            self._lora_loaded = True
            logger.info("LoRA weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load LoRA weights: {e}")
            logger.warning("Running with base Flux model only.")
            self._lora_loaded = False

    # ── Pipeline Switching ───────────────────────────────────────────────

    def _unload_all(self):
        """Unload all pipelines to free VRAM before loading a new one."""
        logger.info(f"Unloading active pipeline (was: {self._active_mode})...")

        self._text2img = None
        self._img2img = None
        self._inpaint = None
        self._ip_adapter_loaded = False
        self._active_mode = None

        flush_vram()

    @property
    def is_lora_loaded(self) -> bool:
        return self._lora_loaded

    @property
    def active_mode(self) -> Optional[str]:
        return self._active_mode


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
_manager: Optional[PipelineManager] = None


def get_pipeline_manager() -> PipelineManager:
    """Get or create the global pipeline manager singleton."""
    global _manager
    if _manager is None:
        _manager = PipelineManager()
    return _manager
