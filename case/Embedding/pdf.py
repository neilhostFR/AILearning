from sentence_transformers import SentenceTransformer
import pymupdf
from PIL import Image
import io

model=SentenceTransformer('clip-ViT-B-32')
doc=pymupdf.open('./pdf/Happy-LLM-0727.pdf')
vectors=[]

for page_num in range(len(doc)):
	page=doc[page_num]

	text=page.get_text()
	if text.strip():
		text_vector = model.encode([text])[0]
		vectors.append(('text', text_vector))

	image_list=page.get_images()
	for img_index,img in enumerate(image_list):
		try:
			xref=img[0]
			pix=pymupdf.Pixmap(doc, xref)
			if pix.n - pix.alpha < 4:
				img_data = pix.tobytes("png")
				img_pil = Image.open(io.BytesIO(img_data))

				image_vector = model.encode([img_pil])[0]
				vectors.append(('image', image_vector))

		except Exception as e:
			print(f"处理图片时出错: {e}")
doc.close()

print(vectors)