from io import BytesIO
import json
import os
import subprocess
import threading
import uuid
from queue import Queue
import config

from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
from flask_socketio import SocketIO, emit
from PIL import Image, UnidentifiedImageError
import base64
import glob
import openai
import voice
from VoicetoText import start_voice_to_text
import io
import shutil


result_queue = Queue()
voice_recognition_active = False

load_dotenv(find_dotenv())  # read local .env file
openai.organization = config.OPENAI_ORG_KEY
openai.api_key = config.OPENAI_API_KEY

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    print("GPT API response:", response)
    return response.choices[0].message["content"]

app = Flask(__name__)
socketio = SocketIO(app)

detected_objects_context = None

@app.route('/')
def index():
    return send_from_directory(app.static_folder, "index.html")

from werkzeug.datastructures import FileStorage


@app.route('/get_transcription', methods=['GET'])
def get_transcription():
    if not config.result_queue.empty():
        transcription = config.result_queue.get()
        return jsonify({"status": "success", "transcription": transcription})
    else:
        return jsonify({"status": "success", "transcription": None})

@app.route('/start_voice_recognition', methods=['POST'])
def start_voice_recognition():
    config.voice_recognition_active = True
    threading.Thread(target=start_voice_to_text, args=(config.result_queue,)).start()
    return jsonify(success=True)

@app.route('/stop_voice_recognition', methods=['POST'])
def stop_voice_recognition():
    config.voice_recognition_active = False
    while not config.result_queue.empty():
        try:
            config.result_queue.get_nowait()
        except Queue.queue.Empty:
            continue
    return jsonify(success=True)



@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text')
    print(text)
    if text:
        audio_file = voice.text_to_speech(text)
        return send_file(io.BytesIO(audio_file), mimetype='audio/mpeg')
    else:
        return 'Invalid request', 400
    
@app.route('/get_image/<filename>')
def get_image(filename):
    image_folder = os.path.join("runs", "detect", "from_uploaded")
    return send_from_directory(image_folder, filename)

def move_image_to_static(src_path, dest_folder='static/images'):
    """
    Move image from src_path to dest_folder
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_folder, filename)

    shutil.move(src_path, dest_path)

    return dest_path

@app.route('/chat', methods=['POST'])
def chat():
    global detected_objects_context
    global label
    messages = request.get_json().get('messages')
    temperature = request.get_json().get('temperature', 1)
    print("Request data:", request.get_json())
    processed_image_url_in_static = ""
    image_filename = None
    text_filename = None
    label = None
    # Check if there is an image attached
    base64_image = request.get_json().get('image')
    if base64_image and not detected_objects_context:
        # Decode the base64 string
        image_data = base64.b64decode(base64_image)

        # Create a FileStorage object from the decoded image data
        image = FileStorage(stream=BytesIO(image_data), content_type='image/jpeg')

        # Save the image
        image_id = uuid.uuid4()
        image_filename = f"IMG_{image_id}.jpeg"
        text_filename = f"IMG_{image_id}.txt"
        image_path = os.path.join("inference", "images", "uploaded", image_filename)
        os.makedirs("inference/images/uploaded", exist_ok=True)
        image.save(image_path)
        detect(image_path)
        processed_image_path = os.path.join("runs", "detect", "from_uploaded", image_filename)
        # Move the processed image to static folder
        processed_image_path_in_static = move_image_to_static(processed_image_path)
        processed_image_url_in_static = request.url_root + processed_image_path_in_static
        #response = get_completion_from_messages(messages, temperature=temperature)

    class_names = [
        'INSL-POST-15KV-PORC-TT-F', 'INSL-POST-15KV-PORC-HC-F', 'INSL-POST-25KV-PORC-TT-F',
        'INSL-POST-25KV-PORC-HC-F', 'INSL-POST-35KV-PORC-TT-F', 'INSL-POST-35KV-PORC-HC-F',
        'INSL-POST-35KV-PORC-VC-F', 'INSL-POST-45KV-PORC-TT-F', 'INSL-POST-45KV-POLY-TT-F',
        'INSL-POST-45KV-POLY-HC-F', 'INSL-POST-45KV-POLY-VC-F', 'INSL-POST-45KV-POLY-HC-GB-F',
        'INSL-PIN-15KV-POLY-F', 'INSL-PIN-23KV-PORC-F', 'INSL-PIN-25KV-POLY-F', 'INSL-DE/S-PORC-F',
        'INSL-DE/S-7KV-PORC-F', 'INSL-DE/S-25KV-POLY-F', 'INSL-DE/S-35KV-POLY-F', 'INSL-DE/S-45KV-POLY-F',
        'INSL-SP-SEC-PORC-F', 'INSL-1RACK-SEC-PORC-F', 'CLAMP-AER-CABLE-MD-F',
    ]

    #class_names = [ 'Pole', 'Tag', 'Transformer', 'Light', 'Fuse', 'Down Guy', 'Viper Recloser', 'Riser', 'Arrester', 'Weatherhead', 'Capacitor', 'Trip Saver', 'Switch' ]

    image = ""
    if text_filename:  # Add this condition
        label = os.path.join("runs", "detect", "from_uploaded", "labels", text_filename)
    if image_filename:  # Add this condition
        image = os.path.join("runs", "detect", "from_uploaded", image_filename)

    print(f"label before if:\n\n'''' {label} ''''")

    from collections import Counter

    detected_objects = []
    object_counts = Counter()  # Use Counter to simplify counting objects

    if label in glob.glob(os.path.join("runs", "detect", "from_uploaded", "labels", "*")):
        print(f"label after if:\n\n'''' {label} ''''")
        with open(label, "r") as f:
            label_data = f.readlines()

            for line in label_data:
                class_id = int(line.strip().split()[0])
                class_name = class_names[class_id]
                object_counts[class_name] += 1  # Increment count for each object

            for object_name, count in object_counts.items():
                detected_objects.append(f"\n\n{object_name}: {count}")

            detected_objects_context = f"\nObjects detected: ```{', '.join(detected_objects)}```"
            messages.append({"role": "assistant", "content": detected_objects_context})


    
    elif image in glob.glob(os.path.join("runs", "detect", "from_uploaded", "*")):
        #print(f"image after if:\n\n'''' {image} ''''")
        detected_objects_context = f"\nObjects detected: ```{', '.join(detected_objects)}```"
        messages.append({"role": "assistant", "content": detected_objects_context})

    if detected_objects_context:
        messages.append({"role": "user", "content": "What did you find in the image?"})
        detected_objects_context = None


    #print(f"Messages:\n\n'''' {messages} ''''")
    response = get_completion_from_messages(messages, temperature=temperature)
    return jsonify({"response": response, "image_filename": processed_image_url_in_static})

@app.route('/detect', methods=['POST'])
def detect(image_path):
    # Change the command based on your custom model, source images, and parameters
    detect_command = [
        "python3",
        "detect.py",
        "--weights",
        "yolov7_custom85_insulators_best.pt",
        "--conf",
        "0.5",
        "--img-size",
        "640",
        "--source",
        image_path,
        "--project",
        "runs/detect",
        "--name",
        "from_uploaded",
        "--exist-ok",
        "--no-trace",
        "--save-conf",
        "--save-txt",
    ]

    try:
        result = subprocess.run(detect_command, check=True, text=True, capture_output=True)
        print("YOLO detection output:", result.stdout)
        return jsonify({"status": "success", "output": result.stdout})
    except subprocess.CalledProcessError as error:
        print("YOLO detection error:", error.stderr)
        return jsonify({"status": "error", "message": error.stderr})


if __name__ == '__main__':
    socketio.run(app, debug=True)
