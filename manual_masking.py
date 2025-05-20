import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import os
os.makedirs("output", exist_ok=True)

# Load original image and mask
image = cv2.imread("images/scannable.jpg")
image = cv2.resize(image, (700, 1000))  # Ensure consistent size
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = cv2.imread("images/mask.png", 0)
mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Paper", masked)
cv2.waitKey(0)

gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
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
        # print("Peri: ",peri)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print( "Approx: ",approx)
        if len(approx) == 4:
            docCnt = approx
            break

# Apply perspective transform
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv2.imshow("Paper", paper)
cv2.waitKey(0)
cv2.imshow("Warped", warped)
cv2.waitKey(0)

# Threshold the warped image
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("thresh", thresh)
cv2.waitKey(0)

# Find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []
print(f"[INFO] Total contours found: {len(cnts)}")
# Make a copy of the image to draw contours on
contour_vis = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

# Draw all contours in blue
cv2.drawContours(contour_vis, cnts, -1, (255, 0, 0), 1)

# Display the image with contours
cv2.imshow("All Contours on Threshold", contour_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    # Adjust thresholds as needed for your checkbox dimensions
    if w >= 7 and h >= 7 and area >= 40 :
        questionCnts.append(c)
        cv2.rectangle(paper, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("Debug", paper)
        cv2.waitKey(0)
        # print("Area: ",area)

# Sort the question contours top-to-bottom
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]

# Define the answer key
ANSWER_KEY = {
    0: 1, 1: 4, 2: 0, 3: 3, 4: 1,
    5: 2, 6: 1, 7: 3, 8: 0, 9: 4,
    10: 2, 11: 0, 12: 1, 13: 3, 14: 4,
    15: 0, 16: 2, 17: 4, 18: 1, 19: 3,
    20: 1, 21: 2, 22: 0, 23: 3, 24: 4,
    25: 2, 26: 0, 27: 4, 28: 1, 29: 3,
    30: 1, 31: 2, 32: 3, 33: 0, 34: 4,
    35: 2, 36: 1, 37: 4, 38: 0, 39: 3,
    40: 2, 41: 0, 42: 3, 43: 1, 44: 4,
    45: 1, 46: 3, 47: 2, 48: 0, 49: 4
}


correct = 0

# Loop over the questions
for (q, i) in enumerate(range(0, len(questionCnts), 5)):
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None
    print("Question: ", q+1)
    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        cv2.imshow("sim", cv2.bitwise_and(thresh, thresh, mask=mask))
        cv2.waitKey(0)
        print(j,total)

        if (bubbled is None or total > bubbled[0]) and total > 300:
            bubbled = (total, j)
    if bubbled is None:
        print("No bubble detected for question ", q+1)
        bubbled = (0, -1)
        continue
    print("Final Choice: ", bubbled[1])

    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    print("Answer: ",k)
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    cv2.imshow("Sheet", paper)
    cv2.waitKey(0)


# Calculate the score
score = (correct / len(ANSWER_KEY)) * 100
cv2.putText(
        paper,
        f"Score: {score}%",
        (10, 30),                          # position
        cv2.FONT_HERSHEY_SIMPLEX,          # font
        1.0,                               # scale
        (255,0,0),                         # color (blue BGR)
        2                                  # thickness
    )

# 6. Encode the annotated image to PNG bytes, then base64
cv2.imshow("Result", paper)
cv2.waitKey(0)
# cv2.imwrite("output/marked_sheet.png", paper)

print(f"Score: {score}%")
