import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageDraw
import requests
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")

# Azure clients initialization
cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))
openai_client = AzureOpenAI(azure_endpoint=azure_oai_endpoint, api_key=azure_oai_key, api_version="2024-02-15-preview")

# Tkinter interface
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        analyze_image(file_path)

def analyze_image(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    try:
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS, VisualFeatures.OBJECTS]
        )

        # Get caption and tags for AI story
        caption = result.caption.text if result.caption else "No caption found."
        tags = ", ".join([tag.name for tag in result.tags.list]) if result.tags else "No tags found."

        story = generate_story(caption, tags)
        
        # Show result in the interface
        result_label.config(text=f"Caption: {caption}\nTags: {tags}\n\nGenerated Story: {story}")

        # Draw objects in the image
        if result.objects:
            draw_objects(image_path, result.objects.list)

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

def draw_objects(image_filename, objects):
    image = Image.open(image_filename)
    draw = ImageDraw.Draw(image)
    color = 'cyan'

    for detected_object in objects:
        r = detected_object.bounding_box
        bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
        draw.rectangle(bounding_box, outline=color, width=3)
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def generate_story(caption, tags):
    system_message = f"You are an AI that creates stories from images. Here is an image with the caption: '{caption}' and tags: {tags}. Generate a short story based on this information."
    
    response = openai_client.chat.completions.create(
        model=azure_oai_deployment,
        temperature=0.7,
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Generate a story from the image data."}
        ]
    )
    
    return response.choices[0].message.content

# Tkinter interface setup
root = Tk()
root.title("AI Image Story Generator")

open_button = Button(root, text="Open Image", command=open_image)
open_button.pack(pady=20)

result_label = Label(root, text="Upload an image to generate a story.", wraplength=400)
result_label.pack(pady=20)

root.mainloop()