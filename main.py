import os
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
from onnxInference import ONNXINFERENCE

# Paso 1: Leer el PDF y convertir cada página a una imagen
def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def draw_bboxes(images, results, class_names):
    # Definir colores para cada clase
    # Añadir más colores según sea necesario
    for i, image in enumerate(images):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # Usar una fuente predeterminada proporcionada por PIL
        boxes, scores, class_ids = results[i]
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cls = int(class_ids[j])
            conf = scores[j]
            class_name = class_names[cls]
            confidence = f"{conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{class_name} {confidence}", fill="red", font=font)
    return images

# Paso 5: Convertir las imágenes con bounding boxes de nuevo a un PDF
def images_to_pdf(images, output_pdf_path):
    images[0].save(output_pdf_path, save_all=True, append_images=images[1:])

# Paso 6: Guardar el PDF resultante y los resultados de la inferencia en un archivo JSON
def main(pdf_path, output_pdf_path, json_output_path):
    # Cargar el modelo YOLOv8 ONNX
    model = ONNXINFERENCE("models/faces.onnx", conf_thres=0.5, iou_thres=0.5)

    # Leer el PDF y convertir a imágenes
    images = pdf_to_images(pdf_path)
    
    results = []
    json_results = []
    class_names = ["claudia", "xochitl", "maynez"]
    for page_num, image in enumerate(images):
        # Convertir la imagen de PIL a un array de NumPy
        image_np = np.array(image)
        # Realizar inferencia
        boxes, scores, class_ids = model(image_np)
        results.append((boxes, scores, class_ids))
        
        # Guardar los resultados en la estructura JSON
        for j, box in enumerate(boxes):
            result = {
                "page": page_num + 1,
                "box": box.tolist(),
                "score": float(scores[j]),
                "class_id": int(class_ids[j]),
                "class_name": class_names[int(class_ids[j])]
            }
            json_results.append(result)

    # Dibujar las bounding boxes, clases y confianzas
    images_with_bboxes = draw_bboxes(images, results, class_names)

    # Convertir las imágenes a PDF
    images_to_pdf(images_with_bboxes, output_pdf_path)

    # Guardar los resultados de la inferencia en un archivo JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(json_results, json_file, indent=4)

# Ejecución del script
if __name__ == "__main__":
    pdf_path = "vertigo.pdf"
    output_pdf_path = "vertigo_bboxs.pdf"
    json_output_path = "inference_results.json"
    main(pdf_path, output_pdf_path, json_output_path)