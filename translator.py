import cv2
import PIL
from PIL import Image
import pytesseract
import googletrans
from googletrans import Translator

translator = Translator()

pytesseract.pytesseract.tesseract_cmd = r'X:/Python projects/MoreXa/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(Image.open('img2.PNG'))
print(f'Распознанный текст: {text}')

source_lang = translator.detect(text).lang
print(f'Распознанный язык: {source_lang}')

translated_text = translator.translate(text, src=source_lang, dest='ru')
print(f'Перевод с {source_lang} на ru:\n{translated_text.text}')