import os
import re
import json
import random
import time
import argparse
import ueberzug.lib.v0 as ueberzug
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm
import caption_processor
import term_image.image
from importlib import reload
import templates

def display_image(image_path):
    image = term_image.image.from_file(image_path)
    image.draw()
#def display_image(image_path):
    #with ueberzug.Canvas() as canvas:
        #image = canvas.create_placement('image', x=0, y=0)
        #image.path = image_path
        #image.visibility = ueberzug.Visibility.VISIBLE
        #time.sleep(10)
        #asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))

def load_model(model_name="Salesforce/blip2-opt-2.7b"):
  global model, processor, device

  print("Loading Model")
  processor = AutoProcessor.from_pretrained(model_name)
  model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

  if torch.cuda.is_available():
    print("CUDA available, using GPU")
    device = "cuda"
  else:
    print("CUDA not available, using CPU")
    device = "cpu"

  print("Moving model to device")
  model.to(device)

def progress_bar(current, total, bar_length = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))

    print('Image {} of {} [{}] {:.0f}%'.format(current, total, arrow + spaces, percent))

def process_template(template, data):
    if hasattr(templates, template):
        template = getattr(templates, template)
    else:
        raise ValueError(f"Template {args.template} not found")
    return template(data)

def review_images(directory, template=None):
    os.chdir(directory)
    images = [f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".png")]
    total_images = len(images)
    for i, filename in enumerate(images, 1):
        display_image(filename)
        progress_bar(i, total_images)
        json_filename = f"{os.path.splitext(filename)[0]}.json"
        if os.path.isfile(json_filename):
            with open(json_filename, 'r') as file:
                data = json.load(file)
                for key, value in data.items():
                    print(f"{key}: {value}")
                if template:
                    processed = process_template(template, data)
                    print(processed)
                    with open(f"{os.path.splitext(filename)[0]}.txt", 'w') as outfile:
                        outfile.write(processed)
                input("Press Enter to continue...")

def tag_images(directory_path, question, tag=None, skip=False):
    os.chdir(directory_path)
    images = [f for f in os.listdir(directory_path) if f.endswith(".jpg") or f.endswith(".png")]
    if skip and tag is None:
        raise ValueError("--skip option requires a --tag value")

    if skip:
        images = [filename for filename in images if not (os.path.isfile(f"{os.path.splitext(filename)[0]}.json") and tag in json.load(open(f"{os.path.splitext(filename)[0]}.json", 'r')))]

    total_images = len(images)
    ai_answer = ""
    for i, filename in enumerate(images, 1):
        image = Image.open(filename)
        try:
            ai_answer = caption_processor.CaptionProcessor(model, processor, device).ask(question, image)
        except:
            print("Error creating caption for file: " + filename)
        display_image(filename)
        progress_bar(i, total_images)
        user_input = input(f'Q: {question}\nAI: {ai_answer}\n?> ')
        answer = ai_answer if user_input == '' else user_input

        json_filename = f"{os.path.splitext(filename)[0]}.json"
        key = tag if tag else '{:02x}'.format(random.randint(0,255))

        if os.path.isfile(json_filename):
            with open(json_filename, 'r') as file:
                data = json.load(file)
            if key in data:
                old_answer = data[key]
                overwrite = input(f"Overwrite existing value of {old_answer}? [Y/n]")
                if overwrite.lower() != 'y':
                    continue
            data[key] = answer
        else:
            data = {key: answer}

        with open(json_filename, 'w') as file:
            json.dump(data, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tag images for machine learning.')
    parser.add_argument('directory', type=str, help='Directory containing images')
    parser.add_argument('question', type=str, help='Question for tagging')
    parser.add_argument('--tag', type=str, default=None, help='Optional tag for key')
    parser.add_argument('--skip', action='store_true', help='Skip images that already have a tag value')
    parser.add_argument('--review', action='store_true', help='Review images that already have a tag value')
    parser.add_argument('--template', type=str, default=None, help='Templating function for output text files')
    args = parser.parse_args()

    try:
        if args.review:
            review_images(args.directory, args.template)
        else:
            load_model()
            tag_images(args.directory, args.question, args.tag, args.skip)
    except KeyboardInterrupt:
        print("Interrupted! Exiting gracefully...")
