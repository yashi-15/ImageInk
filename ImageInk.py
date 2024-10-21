from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

# Import namespaces
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


def main():
    global cv_client
    global client
    global azure_oai_deployment

    try:
        # Get Configuration Settings
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
                azure_endpoint = azure_oai_endpoint, 
                api_key=azure_oai_key,  
                api_version="2024-02-15-preview"
                )
        
        # Get image
        image_file = 'images/street.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        with open(image_file, "rb") as f:
            image_data = f.read()
        
        # Analyze image
        AnalyzeImage(image_file, image_data, cv_client)
        

    except Exception as ex:
        print(ex)


def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,],
        )
        
    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")


    # Get image captions
    if result.caption is not None:
        print("\nCaption:")
        print(" Caption: '{}' ".format(result.caption.text))

    # Get image tags
    if result.tags is not None:
        print("\nTags:")
        tags = ", ".join(tag.name for tag in result.tags.list)
        print(tags)

    # Get objects in the image
    if result.objects is not None:
        print("\nObjects in image:")

        # Prepare image for drawing
        image = Image.open(image_filename)
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        objects = ", ".join(detected_object.tags[0].name for detected_object in result.objects.list)
        print(objects)

    # Generate story
    GenerateStory(result.caption.text, objects, tags)


def GenerateStory(caption, objects, tags):
    # Create a system message
    system_message = """I am a creative storyteller named Forest who weaves captivating stories based on the image captions and objects you provide. 
        Share any caption and objects in the image, and I will craft a unique story inspired by it. 
        Each story will be imaginative, engaging, and will incorporate the essence of the caption to bring it to life. 
        If no caption is provided, I will create a whimsical tale based on nature or adventure, ensuring an element of surprise in every story."""

    # Initialize messages array
    messages_array = [{"role": "system", "content": system_message}]
    input_prompt = "Caption: " + caption + ", Tags: " + tags + ", and Objects: " + objects + " are extracted from an image. Generate an interesting story from the caption, tags and objects of a image."
    print(input_prompt)
    messages_array.append({"role": "user", "content": input_prompt})

    response = client.chat.completions.create(
        model=azure_oai_deployment,
        temperature=0.7,
        max_tokens=1200,
        messages=messages_array
    )
    generated_text = response.choices[0].message.content
    # Add generated text to messages array
    messages_array.append({"role": "assistant", "content": generated_text})

    # Print generated text
    print("Story: " + generated_text + "\n")


if __name__ == "__main__":
    main()