from PIL import Image
import pytesseract

if __name__=="__main__":
	try:
		image_path="./disney_knowledge_base/images/2-万圣节.jpeg"
		image = Image.open(image_path)

		ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
		print(ocr_text)
	except Exception as e:
		print(f"处理图片失败 {image_path}: {e}")