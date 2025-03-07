import cv2
import numpy as np
from paddleocr import PaddleOCR


image_path = "photo_s_wb.PNG"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

processed_image_path = "processed_photo.png"
cv2.imwrite(processed_image_path, gray)

ocr = PaddleOCR(lang='ru')
results = ocr.ocr(processed_image_path, cls=True)

sorted_results = sorted(results[0], key=lambda x: (x[0][0][1], x[0][0][0]))

final_text = " ".join([res[1][0] for res in sorted_results])

print("📜 Распознанный текст:")
print(final_text)
