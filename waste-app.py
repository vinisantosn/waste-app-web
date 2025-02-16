from flask import Flask, render_template, request, jsonify
import cv2
import torch
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)

# Carregar o modelo YOLO
model = YOLO("food_waste_yolo.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Processar a imagem com YOLO
    results = model(frame)
    detections = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        confidence = result.conf[0].item()
        class_id = int(result.cls[0].item())
        label = f"{model.names[class_id]}: {confidence:.2f}"

        detections.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # Desenhar bounding boxes na imagem
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Converter a imagem processada para Base64 para exibição no frontend
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"detections": detections, "processed_frame": f"data:image/jpeg;base64,{encoded_image}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
