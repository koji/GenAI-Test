from PIL import Image
import base64
import gradio as gr
import io
import ollama

# constants
IMAGE_SIZE = (224, 224)
IMAGE_FORMAT = 'PNG'
MODEL_VISION = 'llama3.2-vision'
MODEL_TEXT = 'llama3.2'
ROLE_USER = 'user'

def run_process(text, image=None):
    try:
        if image is not None:
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB').resize(IMAGE_SIZE)
            # Base64 encoding
            base64_image = base64.b64encode(io.BytesIO(pil_image.save(io.BytesIO(), format=IMAGE_FORMAT)).getvalue()).decode('utf-8')
            result = ollama.chat(
                model=MODEL_VISION,
                messages=[{'role': ROLE_USER, 'content': text, 'images': [base64_image]}]
            )['message']['content']
        else:
            result = ollama.chat(
                model=MODEL_TEXT,
                messages=[{'role': ROLE_USER, 'content': text}]
            )['message']['content']

    except Exception as e:
        print(f"Error during processing: {e}")  # for debugging
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
