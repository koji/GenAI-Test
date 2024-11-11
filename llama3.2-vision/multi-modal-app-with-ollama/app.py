from PIL import Image
import base64
import gradio as gr
import io
import ollama

# Constants
IMAGE_SIZE = (224, 224)
MODEL_VISION = 'llama3.2-vision'
MODEL_TEXT = 'llama3.2'

def run_process(text, image=None):
    def encode_image_to_base64(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    try:
        if image is not None:
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB').resize(IMAGE_SIZE)
            base64_image = encode_image_to_base64(pil_image)
            response = ollama.chat(
                model=MODEL_VISION,
                messages=[{'role': 'user', 'content': text, 'images': [base64_image]}]
            )
            result = response['message']['content']
        else:
            response = ollama.chat(
                model=MODEL_TEXT,
                messages=[{'role': 'user', 'content': text}]
            )
            result = response['message']['content']

    except Exception as e:
        result = f"An error occurred: {str(e)}"
    
    return result

def create_interface():
    return gr.Interface(
        fn=run_process,
        inputs=[gr.Textbox(label="", placeholder="Enter your prompt here", lines=4),
                gr.Image(label="Upload an image (optional)")],
        outputs=gr.Textbox(label="Output", lines=5),
        title="Multi-modal app with Ollama",
        description="Test Ollama with text and image inputs using Gradio"
    )

if __name__ == "__main__":
    print("Starting multi-modal app with Ollama!")
    create_interface().launch(debug=True, share=True)
