from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

ANSWER_KEY = {0: 1, 1: 4, 2: 2, 3: 3, 4: 0}  # Example: Q1: B, Q2: E, etc.

def process_bubble_sheet(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_contours = [c for c in contours if cv2.contourArea(c) > 100]
    bubble_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[1])

    selected_answers = {}
    for i, c in enumerate(bubble_contours[:5]):  # Assume 5 questions
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        total = cv2.countNonZero(cv2.bitwise_and(gray, gray, mask=mask))
        selected_answers[i] = total

    filled = {i: k for i, (k, v) in enumerate(sorted(selected_answers.items(), key=lambda item: item[1], reverse=True))}
    
    score = 0
    for q in ANSWER_KEY:
        if filled.get(q) == ANSWER_KEY[q]:
            score += 1

    return {"score": score, "total": len(ANSWER_KEY)}

@app.route("/scan", methods=["POST"])
def scan():
    print("Received request", request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    result = process_bubble_sheet(file.read())
    print(result)
    return jsonify(result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
