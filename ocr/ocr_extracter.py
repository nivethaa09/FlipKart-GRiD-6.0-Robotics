import pytesseract
from PIL import Image
from nlp.text_classifier import nlpresult

def extract_text(image_path):
    #  OCR operation
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    nlptext = nlpresult(text)
    return nlptext

#path="ocr/Test2.jpg"
#print("Extracting text from image...",path)
#a=extract_text(path)

""" 
print("file saved") """