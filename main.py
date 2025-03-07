import cv2
import PIL
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'X:/Python projects/MoreXa/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(Image.open('photo_s_wb.PNG'), lang='rus')
print(f'Распознанный текст:\n{text}')
