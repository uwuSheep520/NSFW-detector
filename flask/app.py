from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the image detection model
image_model = load_model('project_final/module/image.h5')
person_model = load_model('project_final/module/judge_ppl.h5')

# Load the text classification model
text_model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
model_path = "project_final/module/bert_model.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
text_model.load_state_dict(state_dict)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Define image dimensions
img_width, img_height = 224, 224

def preprocess_image(url):
    try:
        response = requests.get(url)
        img = load_img(BytesIO(response.content), target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_person(img_array):
    prediction = person_model.predict(img_array)
    if prediction[0][0] <= 0.5:
        return 1  # Person detected
    else:
        return 0  # No person detected

def predict_image(img_array):
    prediction = image_model.predict(img_array)
    if prediction[0][0] >= 0.9:
        return 1  # NSFW content detected
    else:
        return 0  # No NSFW content detected
    

def prepare_input(sentence):
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > 512:
        tokens = tokens[:512]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids[:512] + [0] * (512 - len(token_ids))
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
    return tokens_tensor

def predict(sentence):
    try:
        text_model.eval()
        with torch.no_grad():
            tokens_tensor = prepare_input(sentence)
            outputs = text_model(tokens_tensor)
            logits = outputs[0]
            probabilities = torch.softmax(logits, dim=1)
            _, predicted_label = torch.max(probabilities, 1)
            return predicted_label.item() == 0
    except Exception as e:
        print(f"Error predicting text: {e}")
        return True  # Assume no NSFW content in case of error

@app.route('/detect_nsfw', methods=['POST'])
def detect_nsfw():
    try:
        data = request.get_json()
        if 'sentence' in data:
            # Text classification
            sentence = data['sentence']
            predicted_label = predict(sentence)
            nsfw_not_detected = predicted_label == 0
            return jsonify({'nsfw_not_detected': nsfw_not_detected})
        elif 'image' in data:
            # Image classification
            image_url = data.get('image')
            img_array = preprocess_image(image_url)
            
            person_judge = predict_person(img_array)
            if person_judge == 1:
                result = predict_image(img_array)
                return jsonify({'result': result})
            else:
                return jsonify({'result': 0})  # No NSFW check if no person detected
        else:
            return jsonify({'error': 'Failed to process image'}), 400
        
    except Exception as e:
        print(f"Error in detect_nsfw endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
