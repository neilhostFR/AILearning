from paddleocr import PaddleOCR

if __name__=="__main__":
	# 创建ocr实例
	ocr = PaddleOCR(
		use_doc_orientation_classify=False,
		use_doc_unwarping=False,
		use_textline_orientation=False
	)

	# 进行ocr识别
	result=ocr.predict("./disney_knowledge_base/images/2-万圣节.jpeg")

	# 结果
	print(result[0]["rec_texts"])
	print("\n".join(result[0]["rec_texts"]))