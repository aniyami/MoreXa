import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'X:/Python projects/MoreXa/Tesseract-OCR/tesseract.exe'

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=40)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    return gray

image_path = 'img2.PNG'
img = preprocess_image(image_path)

custom_config = r'--oem 3 --psm 6 -l eng'
data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
n_boxes = len(data['text'])

recognized_text = ""
lines = []

for i in range(n_boxes):
    if int(data['conf'][i]) > 50 and data['text'][i].strip():
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data['text'][i]
        lines.append((y, x, text, w, h))

sorted_lines = sorted(lines, key=lambda t: (t[0] // 10, t[1]))

line_buffer = []
current_y = -1
final_lines = []

for y, x, text, w, h in sorted_lines:
    if current_y == -1 or abs(y - current_y) > 15:
        if line_buffer:
            final_lines.append(sorted(line_buffer, key=lambda t: t[1]))
        line_buffer = []
        current_y = y

    line_buffer.append((y, x, text, w, h))

if line_buffer:
    final_lines.append(sorted(line_buffer, key=lambda t: t[1]))

for line in final_lines:
    recognized_text += " ".join([word[2] for word in line]) + "\n"

print("Распознанный текст:\n", recognized_text.strip())

img_color = cv2.imread(image_path)

for line in final_lines:
    x_min = min(line, key=lambda t: t[1])[1]
    x_max = max(line, key=lambda t: t[1] + t[3])[1]
    y_min = min(line, key=lambda t: t[0])[0] - 5
    y_max = max(line, key=lambda t: t[0] + t[4])[0] + 5
    cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    for _, x, _, w, h in line:
        cv2.rectangle(img_color, (x, y_min), (x + w, y_max), (0, 0, 255), 1)

cv2.imshow("Result", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
