# 🎬 AI-Powered Meme GIF Generator

> Transform any prompt into hilarious animated memes using cutting-edge AI models!

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

This project implements a comprehensive framework for creating animated GIFs from user prompts using advanced AI models. The system generates humorous and contextually relevant meme content by combining state-of-the-art text generation, image synthesis, and video diffusion technologies.

---

## 🚀 Features

- **🤖 AI-Powered Text Generation**: Uses Llama 3.2 (3B-Instruct) for witty meme captions
- **🎯 Smart Template Matching**: Semantic similarity search for optimal meme template selection  
- **🎥 Video Synthesis**: Stable Video Diffusion for smooth, high-quality GIF generation
- **✨ Dynamic Text Overlay**: Intelligent text positioning with customizable styling
- **📊 Batch Processing**: Generate multiple memes efficiently
- **🔧 Configurable Pipeline**: Easy customization of models and parameters

---

## 🏗️ Architecture Overview

Our pipeline consists of five main components:

1. **User Prompt Processing**: Natural language input parsing and preprocessing
2. **Meme Text Generation**: AI-powered caption creation using Llama 3.2
3. **Template Selection**: Semantic matching with pre-existing meme templates
4. **Video Generation**: Frame-by-frame animation using Stable Video Diffusion
5. **Text Overlay**: Dynamic caption positioning and styling

---AI-Powered Meme GIF Generator

This project implements a framework for creating GIFs from user prompts using advanced AI models. The generated GIFs use humorous or insightful captions to create meme-like content based on the user’s input.

---

## Overview

Our pipeline consists of:
1. **User Prompt**: The user provides a topic or input idea.
2. **Text Generation**: An NLP model, Llama 3.2 (3B - instructed), generates a meme-like caption for the provided prompt.
3. **Meme Template Selection**: A meme template is chosen from a pre-existing database (e.g., Imgflip).
4. **GIF Generation**: Using stable video diffusion, 14 frames are created to form a looping video.
5. **Text Overlay System**: The caption is overlaid onto the generated frames to finalize the GIF.

---

## 📊 Pipeline Visualization

### System Architecture
The MemeCraft framework demonstrates our end-to-end approach from prompt to meme:

![MemeCraft Framework](images_for_readme/Screenshot%202025-01-27%20at%2011.23.43.png)

*Figure 1: Complete pipeline showing template selection, caption generation, and text overlay process*

### System Flow Diagram
Our high-level system architecture illustrates the seamless integration of AI models:

![System Scheme](images_for_readme/Screenshot%202025-01-27%20at%2011.23.51.png)

*Figure 2: System architecture from user input to final animated meme generation*

### Example Results and Performance
Real-world examples showcasing the variety and quality of generated content:

![Example Results](images_for_readme/Screenshot%202025-01-27%20at%2011.24.29.png)

*Figure 3: Sample prompts with corresponding generated captions and resulting GIFs*

---

## 🎯 Example Generated Memes

Here are some examples of prompts and the AI-generated animated memes:

### 1. Classic Reaction Meme
![Meme Example 1](images_for_readme/output_gif_3.gif)

### 2. Relatable Situation
![Meme Example 2](images_for_readme/output_gif_17.gif)

### 3. Pop Culture Reference
![Meme Example 3](images_for_readme/output_gif_32.gif)

### 4. Daily Life Humor
![Meme Example 4](images_for_readme/output_gif_47.gif)

### 5. Internet Culture
![Meme Example 5](images_for_readme/output_gif_87.gif)

---

## 📁 Project Structure

```
gif_maker_AI/
├── 📁 src/                          # Source code
│   ├── 📄 __init__.py               # Package initialization
│   ├── 📄 main.py                   # Main pipeline script
│   ├── 📁 core/                     # Core functionality
│   │   ├── 📄 __init__.py
│   │   ├── 📄 meme_generator.py     # Main meme generation logic
│   │   ├── 📄 video_generator.py    # Video diffusion pipeline
│   │   └── 📄 image_generator.py    # Image generation utilities
│   └── 📁 utils/                    # Utility modules
│       ├── 📄 __init__.py
│       ├── 📄 text_overlay.py       # Text positioning and styling
│       ├── 📄 similarity.py         # Text similarity calculations
│       └── 📄 template_search.py    # Template matching algorithms
├── 📁 data/                         # Data files
│   ├── 📄 meme_text_description.json      # Template descriptions
│   ├── 📄 meme_text_description_new.json  # Updated template data
│   └── 📄 meme_templates.zip        # Compressed meme templates
├── 📁 output/                       # Generated content
│   ├── 📁 frames/                   # Individual video frames
│   └── 📁 gifs/                     # Final GIF outputs
├── 📁 config/                       # Configuration files
│   └── 📄 settings.py               # Application settings
├── 📁 images_for_readme/            # Documentation assets
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.py                      # Package setup configuration
└── 📄 README.md                     # This file
```

---

## 🛠️ Technology Stack

### AI Models
- **🧠 Text Generation**: [Llama 3.2 (3B-Instruct)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) - Advanced language model for meme caption generation
- **🔍 Semantic Search**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - Efficient sentence embeddings for template matching
- **🖼️ Image Generation**: [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) - High-quality image synthesis
- **🎬 Video Diffusion**: [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) - Image-to-video generation

### Core Technologies
- **PyTorch** - Deep learning framework
- **Diffusers** - State-of-the-art diffusion models
- **Transformers** - Pre-trained transformer models
- **Sentence Transformers** - Semantic text similarity
- **FAISS** - Efficient similarity search
- **Pillow** - Image processing and manipulation

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Roy-Ayalon/gif_maker_AI.git
cd gif_maker_AI

# Install dependencies
pip install -r requirements.txt

# Alternative: Install as a package
pip install -e .
```

### 2. Configuration

Create your environment file or update `config/settings.py`:

```python
# Required API Keys
GENAI_API_KEY = "your_google_generative_ai_api_key"
HUGGINGFACE_TOKEN = "your_huggingface_token"
```

### 3. Basic Usage

```bash
# Run the main pipeline
python src/main.py

# Or run individual components
python src/core/meme_generator.py --prompt "Monday mornings"
```

### 4. Advanced Usage

```python
from src.core.meme_generator import MemeGenerator

# Initialize the generator
generator = MemeGenerator()

# Generate a single meme
gif_path = generator.generate_meme("coffee addiction")

# Batch generation
prompts = ["Monday blues", "WiFi issues", "Cat videos"]
results = generator.batch_generate(prompts)
```

---

## 🔧 Configuration

### Model Parameters

You can customize the generation process by modifying `config/settings.py`:

```python
# Generation Parameters
DEFAULT_NUM_FRAMES = 14              # Number of video frames
DEFAULT_INFERENCE_STEPS = 100        # Diffusion steps (quality vs speed)
DEFAULT_HEIGHT = 512                 # Output height
DEFAULT_WIDTH = 512                  # Output width
DEFAULT_GIF_DURATION = 200          # Frame duration in milliseconds

# Text Overlay Settings
DEFAULT_FONT_SIZE = 30               # Base font size
DEFAULT_OUTLINE_THICKNESS = 1        # Text outline thickness
DEFAULT_TEXT_COLOR = (255, 255, 255) # White text
DEFAULT_OUTLINE_COLOR = (0, 0, 0)    # Black outline
```

### Environment Variables

For production deployment, use environment variables:

```bash
export GENAI_API_KEY="your_api_key"
export HUGGINGFACE_TOKEN="your_token"
export CUDA_VISIBLE_DEVICES="0"  # GPU selection
```

---

## 📊 Performance & Optimization

### Memory Usage
- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB VRAM
- **Optimal**: 32GB RAM, 12GB+ VRAM

### Speed Benchmarks
| Component | CPU (avg) | GPU (avg) |
|-----------|-----------|-----------|
| Text Generation | 2-3s | 0.5-1s |
| Template Matching | 0.1-0.2s | 0.05s |
| Video Generation | 60-120s | 15-30s |
| Text Overlay | 0.5-1s | 0.1s |

### Optimization Tips
- Enable model CPU offloading for limited VRAM
- Use mixed precision (fp16) for faster inference
- Batch process multiple prompts for efficiency
- Cache template embeddings for repeated use

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/gif_maker_AI.git

# Create a development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8 and use `black` for formatting
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **Commits**: Use conventional commit messages

### Areas for Contribution

- 🎨 New meme templates and styles
- 🚀 Performance optimizations
- 🧪 Additional AI models integration
- 📱 Web interface development
- 🐛 Bug fixes and improvements

---

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory Error**
```bash
# Solution 1: Enable CPU offloading
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Solution 2: Reduce batch size and resolution
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
```

**Model Download Issues**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Re-login to Hugging Face
huggingface-cli login
```

**Slow Generation Speed**
- Ensure CUDA is properly installed
- Use GPU-optimized models
- Enable xFormers memory efficient attention

### FAQ

**Q: Can I use custom meme templates?**
A: Yes! Add your templates to `data/` and update the JSON descriptions.

**Q: How do I change the text generation model?**
A: Modify `TEXT_GENERATION_MODEL` in `config/settings.py`.

**Q: Can I run this on CPU only?**
A: Yes, but expect significantly slower generation times (10-20x slower).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Roy Ayalon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

### Models and Frameworks
- **Meta AI** - Llama 3.2 language model
- **Stability AI** - Stable Diffusion and Video Diffusion models
- **Hugging Face** - Model hosting and transformers library
- **Google** - Generative AI APIs and tools

### Datasets and Resources
- **Imgflip** - Meme template database
- **Internet Meme Community** - Template descriptions and cultural context

### Development Tools
- **PyTorch Team** - Deep learning framework
- **Diffusers Team** - Diffusion model implementations
- **Open Source Community** - Various utility libraries

---

## 📈 Roadmap

### Version 1.1 (Coming Soon)
- [ ] Web interface with real-time preview
- [ ] Custom font upload support
- [ ] Batch processing with progress tracking
- [ ] Advanced text positioning algorithms

### Version 1.2 (Future)
- [ ] Voice-to-meme generation
- [ ] Multi-language support
- [ ] Social media integration
- [ ] Community template sharing

### Version 2.0 (Long-term)
- [ ] Real-time meme generation
- [ ] AI-powered meme trend analysis
- [ ] Custom model fine-tuning
- [ ] Mobile app development

---

## 📞 Support

- 🐛 **Bug Reports**: [Create an issue](https://github.com/Roy-Ayalon/gif_maker_AI/issues)
- 💡 **Feature Requests**: [Discussion board](https://github.com/Roy-Ayalon/gif_maker_AI/discussions)
- 📧 **Email**: contact@royayalon.dev
- 💬 **Discord**: [Join our community](#)

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Roy-Ayalon/gif_maker_AI&type=Date)](https://star-history.com/#Roy-Ayalon/gif_maker_AI&Date)

---

<div align="center">

**Made with ❤️ by [Roy Ayalon](https://github.com/Roy-Ayalon)**

*Bringing AI-powered humor to the world, one meme at a time!* 🎭

[⬆ Back to Top](#-ai-powered-meme-gif-generator)

</div>