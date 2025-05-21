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

def manual_mask(image_bytes):
    # Convert bytes to OpenCV image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours and locate the document
    def locate_document(img):
        cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        docCnt = None

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                peri = cv2.arcLength(c, True)
                # print("Peri: ",peri)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # print( "Approx: ",approx)
                if len(approx) == 4:
                    docCnt = approx
                    break
        return docCnt

    # Apply perspective transform
    documentCnt = locate_document(edged)
    paper = four_point_transform(image, documentCnt.reshape(4, 2))
    paper = cv2.resize(paper, (400, 600))  
    # warped = four_point_transform(gray, documentCnt.reshape(4, 2))
    # cv2.imshow("Paper", paper)
    # cv2.waitKey(0)

    def mark_sheet():
        # Masking
        mask_img = cv2.imread("images/mask.png", 0)
        mask_img = cv2.resize(mask_img, (paper.shape[1], paper.shape[0]))
        masked = cv2.bitwise_and(paper, paper, mask=mask_img)
        gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blurred_masked = cv2.GaussianBlur(gray_masked, (5, 5), 0)
        edged_masked = cv2.Canny(blurred_masked, 75, 200)
        answerCnt = locate_document(edged_masked)
        paper_crop = four_point_transform(masked, answerCnt.reshape(4, 2))
        warped_crop = four_point_transform(gray_masked, answerCnt.reshape(4, 2))
        # cv2.imshow("Edm", edged_masked)
        # cv2.waitKey(0)
        # cv2.imshow("PaperC", paper_crop)
        # cv2.waitKey(0)
        # cv2.imshow("WarpedC", warped_crop)
        # cv2.waitKey(0)

        # Contourize
        cheat_sheet = cv2.imread("images/contourizer.png")
        cheat_sheet = cv2.resize(cheat_sheet, (paper.shape[1], paper.shape[0]))
        gray_cheat_sheet = cv2.cvtColor(cheat_sheet, cv2.COLOR_BGR2GRAY)
        warped_cheat_sheet = four_point_transform(gray_cheat_sheet, answerCnt.reshape(4, 2))
        # cv2.imshow("Cheet Sheet", warped_cheat_sheet)
        # cv2.waitKey(0)


        # Threshold the warped image
        thresh = cv2.threshold(warped_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh_cheat_sheet = cv2.threshold(warped_cheat_sheet, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # cv2.imshow("threshcheat", thresh_cheat_sheet)
        # cv2.waitKey(0)

        # Find contours in the thresholded image
        cnts = cv2.findContours(thresh_cheat_sheet.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        # print(f"[INFO] Total contours found: {len(cnts)}")

        # # Make a copy of the image to draw contours on
        # contour_vis = cv2.cvtColor(thresh_cheat_sheet.copy(), cv2.COLOR_GRAY2BGR)

        # # Draw all contours in blue
        # cv2.drawContours(contour_vis, cnts, -1, (255, 0, 0), 1)

        # # Display the image with contours
        # cv2.imshow("All Contours on Threshold", contour_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Adjust thresholds as needed for your checkbox dimensions
            # if w >= 5  and h >= 5 and area >= 100 :
            questionCnts.append(c)
                # cv2.rectangle(paper_crop, (x, y), (x + w, y + h), (255,0, 0), 1)
                # cv2.imshow("Debug", paper_crop)
                # cv2.waitKey(0)

        # Sort the question contours top-to-bottom
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        # Define the answer key
        ANSWER_KEY = {
        0: 0, 1: 1, 2: 4, 3: 0, 4: 2,
        5: 1, 6: 4, 7: 3, 8: 2, 9: 2,
        10: 2, 11: 1, 12: 2, 13: 1, 14: 0,
        15: 1, 16: 1, 17: 3, 18: 3, 19: 4,
    }

        correct = 0
        choices = {}


        # Loop over the questions
        for (q, i) in enumerate(range(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None
            # print("Question: ", q+1)
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                # cv2.imshow("sim", cv2.bitwise_and(thresh, thresh, mask=mask))
                # cv2.waitKey(0)
                # print(j,total)

                if (bubbled is None or total > bubbled[0]):
                    bubbled = (total, j)
            if bubbled is None:
                print("No bubble detected for question ", q+1)
                bubbled = (0, -1)
                continue
            # print("Final Choice: ", bubbled[1])
            choices[q] = bubbled[1]

            color = (0, 0, 255)
            k = ANSWER_KEY[q]
            # print("Answer: ",k)
            if k == bubbled[1]:
                color = (0, 255, 0)
                correct += 1

            cv2.drawContours(paper_crop, [cnts[k]], -1, color, 2)

        paper_crop = cv2.resize(paper_crop, (300,500))
        # Encode the annotated image to PNG bytes, then base64
        success, png = cv2.imencode('.png', paper_crop)
        if not success:
            return {"error": "Failed to encode ID image"}
        b64_img_marked = base64.b64encode(png.tobytes()).decode('utf-8')

        # Calculate the score
        score = (correct / len(ANSWER_KEY)) * 100
        cv2.putText(
                paper,
                f"Score: {score}%",
                (10, 590),                          # position
                cv2.FONT_HERSHEY_SIMPLEX,          # font
                0.5,                               # scale
                (255,0,0),                         # color (blue BGR)
                2                                  # thickness
            )

       
        return score, b64_img_marked, correct, len(ANSWER_KEY), choices

    def get_index_no():
        # Masking
        mask_img = cv2.imread("images/mask2.png", 0)
        mask_img = cv2.resize(mask_img, (paper.shape[1], paper.shape[0]))
        masked = cv2.bitwise_and(paper, paper, mask=mask_img)
        gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blurred_masked = cv2.GaussianBlur(gray_masked, (5, 5), 0)
        edged_masked = cv2.Canny(blurred_masked, 75, 200)
        valCnt = locate_document(edged_masked)
        paper_crop = four_point_transform(masked, valCnt.reshape(4, 2))
        warped_crop = four_point_transform(gray_masked, valCnt.reshape(4, 2))
        # cv2.imshow("Edm", edged_masked)
        # cv2.waitKey(0)
        # cv2.imshow("PaperC", paper_crop)
        # cv2.waitKey(0)
        # cv2.imshow("WarpedC", warped_crop)
        # cv2.waitKey(0)

        # Contourize
        cheat_sheet = cv2.imread("images/id_contour.png")
        cheat_sheet = cv2.resize(cheat_sheet, (paper.shape[1], paper.shape[0]))
        gray_cheat_sheet = cv2.cvtColor(cheat_sheet, cv2.COLOR_BGR2GRAY)
        warped_cheat_sheet = four_point_transform(gray_cheat_sheet, valCnt.reshape(4, 2))
        # cv2.imshow("Cheet Sheet", warped_cheat_sheet)
        # cv2.waitKey(0)


        # Threshold the warped image
        thresh = cv2.threshold(warped_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh_cheat_sheet = cv2.threshold(warped_cheat_sheet, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # cv2.imshow("threshcheat", thresh_cheat_sheet)
        # cv2.waitKey(0)

        # Find contours in the thresholded image
        cnts = cv2.findContours(thresh_cheat_sheet.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digitCnts = []
        print(f"[INFO] Total contours found: {len(cnts)}")

        # # Make a copy of the image to draw contours on
        # contour_vis = cv2.cvtColor(thresh_cheat_sheet.copy(), cv2.COLOR_GRAY2BGR)

        # # Draw all contours in blue
        # cv2.drawContours(contour_vis, cnts, -1, (255, 0, 0), 1)

        # # Display the image with contours
        # cv2.imshow("All Contours on Threshold", contour_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Adjust thresholds as needed for your checkbox dimensions
            if w >= 5  and h >= 5 and area >= 90 :
                digitCnts.append(c)
                # cv2.rectangle(paper_crop, (x, y), (x + w, y + h), (255,0, 0), 1)
                # cv2.imshow("Debug", paper_crop)
                # cv2.waitKey(0)


        # Sort the sections contours top-to-bottom
        digitCnts = contours.sort_contours(digitCnts, method="top-to-bottom")[0]
        index_number = ""

        # Loop over the sections
        for (q, i) in enumerate(range(0, len(digitCnts), 6)):
            cnts = contours.sort_contours(digitCnts[i:i + 6])[0]
            bubbled = None
            # print("Digit: ", q+1)
            for (j, c) in enumerate(cnts):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                # cv2.imshow("sim", cv2.bitwise_and(thresh, thresh, mask=mask))
                # cv2.waitKey(0)
                # print(j,total)

                if (bubbled is None or total > bubbled[0]):
                    bubbled = (total, j)
            if bubbled is None:
                print("No bubble detected for question ", q+1)
                bubbled = (0, -1)
                continue
            # print("Final Choice: ", bubbled[1])
            index_number += str((bubbled[1]+1)%6)
            cv2.drawContours(paper_crop, [cnts[bubbled[1]]], -1, (0,255,0), 2)
       
        paper_crop = cv2.resize(paper_crop, (300,500))
        # Encode the annotated image to PNG bytes, then base64
        success, png = cv2.imencode('.png', paper_crop)
        if not success:
            return {"error": "Failed to encode ID image"}
        b64_img_id = base64.b64encode(png.tobytes()).decode('utf-8')

        cv2.putText(
            paper,
            f"ID: {index_number}",
            (10, 570),                          # position
            cv2.FONT_HERSHEY_SIMPLEX,          # font
            0.5,                               # scale
            (255,0,0),                         # color (blue BGR)
            2                                  # thickness
        )

        return int(index_number), b64_img_id


    score_percentage, b64_markedimg, score, total_questions, choices = mark_sheet()
    id, b64_idimg  = get_index_no()
    
    # Encode the annotated image to PNG bytes, then base64
    success, png = cv2.imencode('.png', paper)
    if not success:
        return {"error": "Failed to encode image"}
    b64img = base64.b64encode(png.tobytes()).decode('utf-8')
   
    


    return {"score": score,
            "ID": id,
            "score_percentage": score_percentage,
            "total_questions": total_questions,
            "choices": choices,
            "annotated_image": f"data:image/png;base64,{b64img}",
            "annotated_image_id": f"data:image/png;base64,{b64_idimg}",
            "annotated_image_marked": f"data:image/png;base64,{b64_markedimg}",
            }


@app.route("/scan", methods=["POST"])
def scan():
    print("Received request", request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    # result = process_bubble_sheet(file.read())
    result = manual_mask(file.read())
    return jsonify(result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
