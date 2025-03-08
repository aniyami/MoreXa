import cv2
import numpy as np
from paddleocr import PaddleOCR

image_path = "test_img_code.PNG"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Улучшаем контраст с CLAHE (лучше, чем простая коррекция alpha/beta)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# Применяем адаптивную бинаризацию
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 15, 5)

# Убираем шум методом NLM
gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

# Сохраняем обработанное изображение
processed_image_path = "processed_photo.png"
cv2.imwrite(processed_image_path, gray)

# Используем PaddleOCR с несколькими языками (русский, английский, китайский, немецкий и др.)
ocr = PaddleOCR(lang="ru|en|de|fr|es|ch")

# Распознаем текст
results = ocr.ocr(processed_image_path, cls=True)

# Сортируем результаты (по вертикальному и горизонтальному положению текста)
sorted_results = sorted(results[0], key=lambda x: (x[0][0][1], x[0][0][0]))

# Объединяем текст в финальный результат
final_text = "\n".join([res[1][0] for res in sorted_results])

print("📜 Распознанный текст:")
print(final_text)
