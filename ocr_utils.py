from PIL import Image
import pytesseract

# Configure your tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image)
    return text