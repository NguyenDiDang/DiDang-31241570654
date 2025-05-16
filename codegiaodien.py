import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
yolo_model = YOLO("yolov8s.pt")

keras_model_path = "final_model.h5"
tflite_model_path = "final_model.tflite"

import os
if not os.path.exists(tflite_model_path):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# Danh sách nhãn món ăn
target_labels = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com trang", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

# Bảng giá món ăn
price_table = {
    "Ca hu kho": 10000,
    "Canh cai": 8000,
    "Canh chua": 8000,
    "Com trang": 5000,
    "Dau hu sot ca": 7000,
    "Ga chien": 12000,
    "Rau muong xao": 6000,
    "Thit kho": 12000,
    "Thit kho trung": 14000,
    "Trung chien": 7000
}

def classify_image(image):
    try:
        results = yolo_model(image)
        detections = results[0].boxes.data.cpu().numpy()

        seen_classes = set()

        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            if score < 0.3:
                continue

            crop = image[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (input_width, input_height))
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_index = int(np.argmax(output_data))
            predicted_label = target_labels[predicted_index]

            seen_classes.add(predicted_label)

        if not seen_classes:
            return "⚠️ Không phát hiện được món ăn", "0đ"

        result_text = "\n".join([f"🍽️ {food}: {price_table[food]:,}đ" for food in sorted(seen_classes)])
        total_price = sum([price_table[food] for food in seen_classes])
        return result_text, f"💰 Tổng cộng: {total_price:,}đ"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}", "0đ"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🍱 Nhận Diện Món Ăn & Tính Tiền Căn Tin
    Tải ảnh khay cơm → Phát hiện món ăn bằng YOLOv8 → Phân loại bằng CNN → Tính tổng hóa đơn.
    """)

    with gr.Row():
        image_input = gr.Image(type="numpy", label="📸 Ảnh khay cơm")

    with gr.Row():
        food_output = gr.Textbox(label="📋 Món ăn và giá tiền", lines=10)
        total_output = gr.Textbox(label="💵 Tổng tiền cần thanh toán")

    btn = gr.Button("🚀 Nhận diện và tính tiền")
    btn.click(fn=classify_image, inputs=image_input, outputs=[food_output, total_output])

    gr.Markdown("---")
    gr.Markdown("<p style='text-align: center;'>© 2025 - Đồ án Trí tuệ nhân tạo UEH</p>")

demo.launch(debug=True)
