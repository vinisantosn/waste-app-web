from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Carregar o modelo YOLO
model = YOLO("food_waste_yolo.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Inicializar a câmera
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue

        # Realizar detecção de objetos em tempo real
        results = model(frame)

        # Desenhar bounding boxes no frame
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0].item()
            class_id = int(result.cls[0].item())
            label = f"{model.names[class_id]}: {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
