from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import os
import base64


app = Flask(__name__)
CORS(app)

ANSWER_KEY = {0: 1, 1: 4, 2: 2, 3: 3, 4: 0}  # Example: Q1: B, Q2: E, etc.

def process_bubble_sheet(image_bytes):
    # Convert bytes to OpenCV image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours and locate the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    # Apply perspective transform
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # Threshold the warped image
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
            questionCnts.append(c)

    # Sort the question contours top-to-bottom
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

    # Define the answer key
    ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

    correct = 0

    # Loop over the questions
    for (q, i) in enumerate(range(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        color = (0, 0, 255)
        k = ANSWER_KEY[q]
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    # Calculate the score
    score = (correct / 5.0) * 100
    
    # verlay score text
    cv2.putText(
        paper,
        f"Score: {score}%",
        (10, 30),                          # position
        cv2.FONT_HERSHEY_SIMPLEX,         # font
        1.0,                               # scale
        (255,0,0),                         # color (blue BGR)
        2                                  # thickness
    )

    # 6. Encode the annotated image to PNG bytes, then base64
    success, png = cv2.imencode('.png', paper)
    if not success:
        return {"error": "Failed to encode image"}
    b64img = base64.b64encode(png.tobytes()).decode('utf-8')
    


    return {"score": score, "total": len(ANSWER_KEY),
            "annotated_image": f"data:image/png;base64,{b64img}"}

@app.route("/scan", methods=["POST"])
def scan():
    print("Received request", request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    result = process_bubble_sheet(file.read())
    return jsonify(result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
