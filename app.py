from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

FACE_CASCADE = cv2.CascadeClassifier(
    "static/haarcascade_frontalface_default.xml"
)

EYE_CASCADE = cv2.CascadeClassifier(
    "static/haarcascade_eye.xml"
)

app = Flask(__name__)

UPLOAD = "static/uploads"
OUTPUT = "static/output"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

original = None
current = None
filename = None
extension = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload", methods=["POST"])
def upload():
    global original, current, filename, extension

    file = request.files["image"]
    filename = secure_filename(file.filename)
    extension = filename.split(".")[-1]

    path = os.path.join(UPLOAD, filename)
    file.save(path)

    original = cv2.imread(path)
    current = original.copy()

    cv2.imwrite(os.path.join(OUTPUT, filename), current)

    return jsonify({"image": filename})


@app.route("/process", methods=["POST"])
def process():
    global current, filename, extension

    data = request.json
    action = data["action"]
    value = data.get("value", 0)

    img = current.copy()

    # -------- Intensity --------
    if action == "brightness":
        img = cv2.convertScaleAbs(img, beta=int(value))

    elif action == "contrast":
        img = cv2.convertScaleAbs(img, alpha=float(value))

    elif action == "negative":
        img = 255 - img

    # -------- Colors --------
    elif action == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif action == "binary":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif action == "red":
        img[:, :, 0] = 0
        img[:, :, 1] = 0

    elif action == "green":
        img[:, :, 0] = 0
        img[:, :, 2] = 0

    elif action == "blue":
        img[:, :, 1] = 0
        img[:, :, 2] = 0

    elif action == "sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        img = cv2.transform(img, kernel)
        img = np.clip(img, 0, 255)

    # -------- Transform --------
    elif action == "flip_h":
        img = cv2.flip(img, 1)

    elif action == "flip_v":
        img = cv2.flip(img, 0)

    elif action == "rotate_90":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif action == "rotate_180":
        img = cv2.rotate(img, cv2.ROTATE_180)

    elif action == "zoom_in":
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 1.2), int(h * 1.2)))

    elif action == "zoom_out":
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)))

    # -------- Conversion --------
    elif action in ["jpg", "png", "webp"]:
        extension = action
        filename = filename.split(".")[0] + "." + extension

    elif action == "reset":
        img = original.copy()

    current = img
    cv2.imwrite(os.path.join(OUTPUT, filename), img)

    return jsonify({"image": filename})


@app.route("/download")
def download():
    return send_file(os.path.join(OUTPUT, filename), as_attachment=True)

FACE_CASCADE = cv2.CascadeClassifier(
    "static/haarcascade_frontalface_default.xml"
)

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    file = request.files["image"]
    filename = secure_filename(file.filename)

    path = os.path.join(UPLOAD, filename)
    file.save(path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    eye_count = 0

    for (x, y, w, h) in faces:
        # Face rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = EYE_CASCADE.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10
        )

        for (ex, ey, ew, eh) in eyes:
            eye_count += 1
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

    output_path = os.path.join(OUTPUT, filename)
    cv2.imwrite(output_path, img)

    return jsonify({
        "faces": len(faces),
        "eyes": eye_count,
        "image": filename
    })


if __name__ == "__main__":
    app.run(debug=True)
