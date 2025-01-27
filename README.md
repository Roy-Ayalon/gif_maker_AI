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
<img width="864" alt="Screenshot 2025-01-27 at 11 23 43" src="https://github.com/user-attachments/assets/1ebeb5b2-0bf4-407c-88d2-a8ced39755ca" />
This image demonstrates how the system selects templates, generates captions, and overlays text.

### 2. System Scheme
<img width="1077" alt="Screenshot 2025-01-27 at 11 23 51" src="https://github.com/user-attachments/assets/dfad3c03-ce10-418e-ae9d-0e6edb3dc6a7" />
This figure shows the high-level steps from user input to the final meme GIF creation.

### 3. Example User Prompts and Results
<img width="1357" alt="Screenshot 2025-01-27 at 11 24 29" src="https://github.com/user-attachments/assets/e0b61f2c-812d-4109-b450-4b0e83c4e785" />
The table provides examples of prompts, the generated captions, and the resulting GIFs.


---

## Models Used

- **NLP Model**: Llama 3.2 (3B - instructed) for generating meme-like text based on user prompts.
- **Image-to-Video Generator**: Stable Video Diffusion for producing smooth, high-quality meme frames.
- **Text Overlay API**: Imgflip API for overlaying captions onto generated GIF frames.

---

## Example Results

Below are some examples of prompts and the generated GIFs:

1. ![output_gif_3](https://github.com/user-attachments/assets/27e0ae26-e1f4-4715-ac34-fee1437031f2)

2.![output_gif_17](https://github.com/user-attachments/assets/01d74a7b-705a-4c44-b1db-ec2c2d8165fd)

3. ![output_gif_32](https://github.com/user-attachments/assets/03a41a2f-7e3a-4610-95e4-e3d070f50f10)

4. ![output_gif_47](https://github.com/user-attachments/assets/5c729d06-5647-42c7-bf51-38dffd0d1cbc)

5. ![output_gif_87](https://github.com/user-attachments/assets/e54443e0-c989-485c-b35a-7d5dec9e028f)
   

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

