# Configuration settings for the AI Meme GIF Generator

# API Keys (Set these as environment variables or update here)
GENAI_API_KEY = "your_google_generative_ai_api_key_here"
HUGGINGFACE_TOKEN = "your_huggingface_token_here"

# Model Configuration
TEXT_GENERATION_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
SENTENCE_ENCODER_MODEL = "all-MiniLM-L6-v2"
IMAGE_GENERATION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VIDEO_GENERATION_MODEL = "stabilityai/stable-video-diffusion-img2vid"

# Generation Parameters
DEFAULT_NUM_FRAMES = 14
DEFAULT_INFERENCE_STEPS = 100
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_GIF_DURATION = 200

# File Paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
FRAMES_DIR = "output/frames"
GIFS_DIR = "output/gifs"
MEME_TEMPLATES_PATH = "data/meme_templates.zip"
MEME_DESCRIPTIONS_PATH = "data/meme_text_description_new.json"

# Text Overlay Settings
DEFAULT_FONT_SIZE = 30
DEFAULT_OUTLINE_THICKNESS = 1
DEFAULT_TEXT_COLOR = (255, 255, 255)  # White
DEFAULT_OUTLINE_COLOR = (0, 0, 0)     # Black
DEFAULT_MARGIN = 2
