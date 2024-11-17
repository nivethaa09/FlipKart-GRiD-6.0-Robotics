import pytesseract
from PIL import Image

def extract_text(image_path):
    #  OCR operation
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

#path="ocr/Test2.jpg"
#print("Extracting text from image...",path)
#a=extract_text(path)

""" with open("ocr/ocr.txt",'w') as ocr_file:
    ocr_file.write(a)
print("file saved") """