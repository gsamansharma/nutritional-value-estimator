import streamlit as st
from PIL import Image
import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("ROBOFLOW_API_KEY")

# Initialize the Roboflow client with the API key from environment variables
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Title for the app
st.title("Food Image Classifier")

# Image upload option
uploaded_file = st.file_uploader("Choose an image of food...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Food Image.", use_column_width=True)

    # Save the image to a temporary location on disk
    image_path = os.path.join("temp_image.jpg")
    image.save(image_path)

    # Inference on the uploaded image
    st.write("Classifying the food image...")
    try:
        # Infer using the image file path
        result = CLIENT.infer(image_path, model_id="mithayi/2")

        # Process the result and extract predictions
        predictions = result.get("predictions", [])
        if predictions:
            st.write("Predictions:")
            for prediction in predictions:
                pred_class = prediction.get("class")
                confidence = prediction.get("confidence")
                st.write(f"- **Class**: {pred_class}, **Confidence**: {confidence:.2f}")
        else:
            st.write("No predictions found.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)
