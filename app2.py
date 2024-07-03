import gradio as gr
import numpy as np
import cv2  # Ensure you have opencv-python installed
from tensorflow.keras.models import load_model  # Ensure you have TensorFlow installed

IMG_SIZE = 128  # Image size for the model input

# Load your trained model
model = load_model(r'breast_cancer_detection_model5.h5')  # Update this path to your actual model file

# Define class names according to your model
class_names = ['benign', 'malignant', 'normal']  # Update this list if different

# Define the prediction function
def predict_cancer(images):
    results = []
    for img in images:
        # Convert image to grayscale (if it's not already), resize, and normalize
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if not already
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        class_idx = np.argmax(prediction[0])
        class_name = class_names[class_idx]
        probability = np.max(prediction[0])
        results.append(f"{class_name} (Probability: {probability:.2f})")
    
    return results

# Define Gradio interface
def classify_images(images):
    if not isinstance(images, list):  # Ensure images is a list of images
        images = [images]
    return predict_cancer(images)

# Define the Gradio interface
input_images = gr.Image(type='numpy', label='Upload Ultrasound Images')
output_labels = gr.Textbox(label='Predictions')

gr_interface = gr.Interface(
    fn=classify_images,
    inputs=input_images,
    outputs=output_labels,
    title="Breast Cancer Detection from Ultrasound Images",
    description="Upload multiple breast ultrasound images to get predictions on whether they show benign, malignant, or normal conditions."
)

# Launch the Gradio app
if __name__ == "__main__":
    gr_interface.launch()
