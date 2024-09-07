import os
import csv
import requests
import json
import pandas as pd
from openai import OpenAI

# Set up API keys
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
OPENAI_MODEL_NAME = "gpt-4o-mini-2024-07-18"

client = OpenAI()

scene_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "SceneDataSchema",
        "description": "Schema for scene data, including voice-over and image prompt information.",
        "schema": {
            "type": "object",
            "properties": {
                "voice_over": {"type": "string"},
                "image_prompt": {"type": "string"},
            },
            "required": ["voice_over", "image_prompt"],
            "additionalProperties": False
        },
        "strict": True  
    }
}



def generate_story_summary(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an excellent narrator and writer."},
            {
                "role": "user",
                "content": f"Write a short interesting summary about this topic: {topic}. Try to be factual if the topic is real",
            },
        ],
    )
    return response.choices[0].message.content



topic = "the mystery of the great pyramid of giza"
summary = generate_story_summary(topic)

n_scenes = 5
messages = [
            {
                "role": "system",
                "content": "You are a youtube content creator expert in creating captivating voice over, and also utilizing AI to generate detailed images using image generation model.",
            },
            {
                "role": "user",
                "content": f"Based on this summary: '{summary}', create {n_scenes} scenes to tell a story or share interesting facts and details. For each scene, provide the voice_over text and an image_prompt in the given structure. The voice_over should should be at most 6 sentences, while the image_prompt should be simple illustration and not too complicated. The scenes should be optimized for short form media",
            },
            {
                "role": "user",
                "content": "start by creating the first scene."
            }
        ]
scene_prompts = []
for i in range(n_scenes+1):
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,
        response_format=scene_schema,
    )
    response_dict = json.loads(response.choices[0].message.content)
    scene_prompts.append((response_dict["voice_over"], response_dict["image_prompt"]))
    messages.append({
        "role": "system",
        "content": f"Scene {i+1}: {response_dict['voice_over']}"
    })
    if i == n_scenes - 1:
        messages.append({
            "role": "user",
            "content": f"Proceed to create the final scene."
        })
    else:
         messages.append({
            "role": "user",
            "content": f"Proceed to create the next scene."
        })


for a,b in scene_prompts:
    print(a)
    print("-"*80)
    
    
i = 1
folder_name = "new_project"
os.makedirs(f"./{folder_name}", exist_ok=True)
for _, p in scene_prompts:
    out = replicate.run(
    "black-forest-labs/flux-schnell",
    input={"prompt": f"An illustration of {p}",
          "output_format": "png",
           "aspect_ratio": "16:9",
          }
    )
    urlretrieve(out[0], f"{folder_name}/aspect-scene-{i}.png")
    i+=1
    
    

df = pd.DataFrame(scene_prompts, columns=["voice_over", "prompt"])
df.to_csv(os.path.join(folder_name, "generated_text.csv"), index=False)
