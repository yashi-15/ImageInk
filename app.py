import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import io

# Initialize global clients
cv_client = None
client = None
azure_oai_deployment = None

# Load environment variables and initialize clients
def initialize_clients():
    global cv_client, client, azure_oai_deployment
    load_dotenv()
    ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")

    # Authenticate Azure AI Vision client
    cv_client = ImageAnalysisClient(
        endpoint=ai_endpoint,
        credential=AzureKeyCredential(ai_key)
    )

    # Initialize the Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version="2024-02-15-preview"
    )

# Analyze the uploaded image and return caption, tags, and objects
def analyze_image(image_data):
    try:
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS
            ],
        )
        
        # Extract results
        caption = result.caption.text if result.caption else ""
        tags = ", ".join(tag.name for tag in result.tags.list) if result.tags else ""
        objects = ", ".join(detected_object.tags[0].name for detected_object in result.objects.list) if result.objects else ""

        return caption, tags, objects

    except HttpResponseError as e:
        st.error(f"Error analyzing the image: {e}")
        return None, None, None

# Generate a story using the caption, tags, and objects
def generate_story(caption, tags, objects):
    system_message = """I am a creative storyteller named Forest who weaves captivating stories based on the image captions and objects you provide.
        Share any caption and objects in the image, and I will craft a unique story inspired by it."""
    
    input_prompt = f"Caption: {caption}, Tags: {tags}, and Objects: {objects} are extracted from an image. Generate an interesting story from the caption, tags, and objects."

    messages_array = [{"role": "system", "content": system_message},
                      {"role": "user", "content": input_prompt}]
    
    try:
        response = client.chat.completions.create(
            model=azure_oai_deployment,
            temperature=0.7,
            max_tokens=1200,
            messages=messages_array
        )
        generated_text = response.choices[0].message.content
        return generated_text

    except Exception as e:
        st.error(f"Error generating story: {e}")
        return ""

# Streamlit app
def main():
    st.title("Image Analysis and Story Generation")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to byte data
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Initialize clients if not done already
        if cv_client is None or client is None:
            initialize_clients()

        # Analyze the image
        st.write("Analyzing the image...")
        caption, tags, objects = analyze_image(img_byte_arr)

        if caption or tags or objects:
            st.write(f"**Caption**: {caption}")
            st.write(f"**Tags**: {tags}")
            st.write(f"**Objects**: {objects}")

            # Generate a story
            if st.button("Generate Story"):
                story = generate_story(caption, tags, objects)
                st.subheader("Generated Story")
                st.write(story)
        else:
            st.error("No caption, tags, or objects were found.")

if __name__ == "__main__":
    main()
