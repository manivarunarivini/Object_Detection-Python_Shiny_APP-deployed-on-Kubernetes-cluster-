import cv2
import numpy as np
import owncloud
import onnxruntime as ort
import os
from datetime import datetime

#Leaf Classes
CLASS_NAMES = [
    "Ivy", "Fern", "Ginkgo", "Kummerowia striata",
    "Laciniata", "Macrolobium acaciifolium",
    "Micranthes odontoloma", "Murraya",
    "Robinia pseudoacacia", "Selaginella davidi franch"
]

SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

PUBLIC_LINK = "https:Link_Protected"
FOLDER_PASSWORD = "Protected"
IMAGE_FOLDER = "RSPi_leaf_images"
PREDICTED_FOLDER = "Predicted_Images"
MODEL_FILENAME = "Yolov8s_Object_Detection_1.onnx"
LAST_PROCESSED_FILE = "last_processed.txt"

class YOLOv8:
    def __init__(self, onnx_model: str, classes: list, confidence_thres=0.5, iou_thres=0.5):
        self.session = ort.InferenceSession(onnx_model)
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.color_palette = np.random.uniform(0, 255, size=(len(classes), 3))
        self.font_thickness = 7
        self.fontscale = 5
        self.img = None
        self.last_class_id = None  # To store the dominant class

    def letterbox(self, img, new_shape=(416, 416)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (top, left)

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 7)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, self.font_thickness)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (int(label_x), int(label_y - label_height)), (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, self.fontscale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def preprocess(self):
        self.img_height, self.img_width = self.img.shape[:2]
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img, pad = self.letterbox(img)
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data, pad

    def postprocess(self, input_image, output, pad):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        gain = min(self.input_height / self.img_height, self.input_width / self.img_height)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        if len(indices) > 0:
            # Store the dominant class (first in indices)
            self.last_class_id = class_ids[indices[0]]
            for i in indices:
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                self.draw_detections(input_image, box, score, class_id)

        return input_image

    def get_predicted_class(self):
        """Returns the clean predicted class name for filenames"""
        if self.last_class_id is None:
            return "Unknown"
        
        class_name = self.classes[self.last_class_id]
        # Handle multi-word class names (take first word)
        return class_name.split()[0]

    def main(self, img):
        self.img = img
        input_shape = self.session.get_inputs()[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        img_data, pad = self.preprocess()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        return self.postprocess(self.img, outputs, pad)

def extract_timestamp(filename):
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            dt_str = parts[-2] + "_" + parts[-1].split('.')[0]
            return datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    except Exception:
        pass
    return None

def extract_timestamp_string(filename):
    """Extracts the timestamp portion from filename"""
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[-2:])  # Returns "YYYYMMDD_HHMMSS.jpg"
    except Exception:
        pass
    return None

def main():
    cloud = owncloud.Client.from_public_link(PUBLIC_LINK, folder_password=FOLDER_PASSWORD)

    with open(MODEL_FILENAME, "wb") as f:
        f.write(cloud.get_file_contents(MODEL_FILENAME))
    model = YOLOv8(MODEL_FILENAME, CLASS_NAMES)

    last_ts = None
    try:
        ts_data = cloud.get_file_contents(f"{PREDICTED_FOLDER}/{LAST_PROCESSED_FILE}").decode().strip()
        last_ts = datetime.strptime(ts_data, "%Y%m%d_%H%M%S")
        print("Last processed:", last_ts)
    except Exception:
        print("[INFO] No last_processed.txt found. Will process all images.")

    files = cloud.list(IMAGE_FOLDER)
    new_images = []
    for f in files:
        name = f.get_name()
        if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            ts = extract_timestamp(name)
            if ts and (last_ts is None or ts > last_ts):
                new_images.append((name, ts))

    new_images.sort(key=lambda x: x[1])
    print(f"[INFO] Found {len(new_images)} new images to process.")

    for name, ts in new_images:
        print(f"[INFO] Processing {name}")
        data = cloud.get_file_contents(f"{IMAGE_FOLDER}/{name}")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        result = model.main(img)

        # Get predicted class name (cleaned for filename)
        predicted_class = model.get_predicted_class()
        
        # Get timestamp portion from original filename
        timestamp_str = extract_timestamp_string(name)
        if timestamp_str is None:
            timestamp_str = ts.strftime("%Y%m%d_%H%M%S") + ".jpg"
        
        # Create new filename
        new_filename = f"{predicted_class}_{timestamp_str}"
        print(f"Saving as: {new_filename}")

        # Save the predicted image
        _, encoded = cv2.imencode(".jpg", result)
        cloud.put_file_contents(f"{PREDICTED_FOLDER}/{new_filename}", encoded.tobytes())

        # Update last processed timestamp
        cloud.put_file_contents(f"{PREDICTED_FOLDER}/{LAST_PROCESSED_FILE}", ts.strftime("%Y%m%d_%H%M%S").encode())

if __name__ == "__main__":
    main()