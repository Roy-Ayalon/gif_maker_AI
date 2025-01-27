# gif_maker_AI

# AI-Powered Meme GIF Generator

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

## Pipeline Explanation

The pipeline flow is visualized in the following figures:

### 1. MemeCraft Framework
[Insert Image Here: Screenshot 2025-01-27 at 11.23.43.png]
This image demonstrates how the system selects templates, generates captions, and overlays text.

### 2. System Scheme
[Insert Image Here: Screenshot 2025-01-27 at 11.23.51.png]
This figure shows the high-level steps from user input to the final meme GIF creation.

### 3. Example User Prompts and Results
[Insert Image Here: Screenshot 2025-01-27 at 11.24.29.png]
The table provides examples of prompts, the generated captions, and the resulting GIFs.

### 4. Output Alignment and Quality
[Insert Image Here: Screenshot 2025-01-27 at 11.24.57.png]
This evaluation chart explains how the generated content aligns with the user’s prompt, template quality, and output fidelity.

---

## Models Used

- **NLP Model**: Llama 3.2 (3B - instructed) for generating meme-like text based on user prompts.
- **Image-to-Video Generator**: Stable Video Diffusion for producing smooth, high-quality meme frames.
- **Text Overlay API**: Imgflip API for overlaying captions onto generated GIF frames.

---

## Example Results

Below are some examples of prompts and the generated GIFs:

1. Prompt: **"Work struggles"**
   - Caption: "When bae says 'I love you' but your boss says 'I love profit'"
   - [Generated GIF Placeholder]

2. Prompt: **"Cooking disasters"**
   - Caption: "When life gives you lemons, make lemonade. When life gives you a cooking disaster, make a fire extinguisher."
   - [Generated GIF Placeholder]

3. Prompt: **"Netflix binging"**
   - Caption: "When Netflix binges, I'm not binge-watching. I'm just casually ignoring my responsibilities for the next 3 days."
   - [Generated GIF Placeholder]

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/AI-Meme-GIF-Generator.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys for accessing the meme template database (e.g., Imgflip API).

---

## Usage

Run the script with a user prompt:
```bash
python generate_meme.py --prompt "Your prompt here"
```
The script will output the generated GIF in the `output/` directory.

---

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- OpenAI for the Llama NLP model.
- Stability AI for Stable Video Diffusion.
- Imgflip for their API and meme template database.

---

### Notes

- Please ensure you have the necessary API keys and access rights to use external services.
- Add examples of generated GIFs into the placeholders above for a complete README.

---

