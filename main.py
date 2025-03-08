import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'X:/Python projects/MoreXa/Tesseract-OCR/tesseract.exe'

img = cv2.imread('photo_s_wb.PNG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
n_boxes = len(data['text'])

recognized_text = ""
last_y = -1
line_buffer = []

for i in range(n_boxes):
    if int(data['conf'][i]) > 60 and data['text'][i].strip() != "":
        current_y = data['top'][i]

        if last_y != -1 and abs(current_y - last_y) > 10:
            recognized_text += " ".join(line_buffer) + "\n"
            line_buffer = []

        line_buffer.append(data['text'][i])
        last_y = current_y

if line_buffer:
    recognized_text += " ".join(line_buffer)

print(recognized_text.strip())

for i in range(n_boxes):
    if int(data['conf'][i]) > 60 and data['text'][i].strip() != "":
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
