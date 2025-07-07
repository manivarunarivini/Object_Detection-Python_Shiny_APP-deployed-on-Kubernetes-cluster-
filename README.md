# Master Thesis: Remote Object Detection system using Raspberry Pi, YOLOv8s, and Python Shiny

This project delivers an automated leaf object detection system combining edge computing, AI inference, and interactive visualization. Images are captured using a Raspberry Pi, processed using a YOLOv8 ONNX model, and visualized through a web-based Python Shiny app. All components are synchronized via a shared Nextcloud folder to ensure seamless data exchange and centralized storage.

## Hardware and Infrastructure
The system uses a Raspberry Pi 4 with a Raspberry Pi HQ camera module for field data acquisition. Captured images, configuration files, predictions, and models are stored in Nextcloud under a structured project folder. The complete application is containerized using a Dockerfile and deployed to a Kubernetes cluster, making the Shiny app publicly accessible via the hosted URL.

## 🔁 Workflow Summary

The Raspberry Pi captures images based on parameters like resolution, measurement interval, and capture_mode, defined in the config.json file.

1. Timed Loop: Continuously captures and uploads images at the configured interval.
2. Single Shot: Captures a single image, then change the mode to "Idle Mode".
3. Idle Mode: Halts image capture and wait for the mode to change. Each image is timestamped using the format _<YYYYMMDD_HHMMSS>.jpg and uploaded to the RSPi_Leaf_Images folder in Nextcloud

* These images are **automatically uploaded** to the `RSPi_leaf_images` folder in **Nextcloud**.
* A YOLOv8 model, exported in **ONNX format**, is trained on **10 leaf classes**:
  ```
  Ivy, Fern, Ginkgo, Kummerowia striata,
  Laciniata, Macrolobium acaciifolium,
  Micranthes odontoloma, Murraya,
  Robinia pseudoacacia, Selaginella davidi franch
  ```

* The script `yolov8_nextcloud_predictor_timestamped.py`:
  * Downloads new images from Nextcloud and performs object detection
  * Renames each image as `<ClassName>_<YYYYMMDD>_<HHMMSS>.jpg`
  * Uploads the result to the `Predicted_Images` folder on nextcloud
  * Updates `last_processed.txt` to track processed images
    
---
## 📦 Components

### ✅ Prediction Script

* `yolov8_nextcloud_predictor_renamed.py`
* Uses OpenCV, ONNX Runtime, and NumPy
* Handles preprocessing, prediction, renaming, and uploading

## YOLOv8s Inference (ONNX)
When the "Refresh Predictions" button is clicked in the Shiny app, the backend triggers a YOLOv8s ONNX model to process newly uploaded images.

Previously analyzed images are skipped using the timestamp recorded in last_predicted.txt.
Results are saved in the Predicted_Images folder in Nextcloud.

The model used for inference was trained on 10 distinct leaf classes using the ultalytics YOLOv8 framework. 

### ✅ Configuration and Visualization (Shiny App)
The Shiny App allows users to modify:

Image capture settings(resolution, measurememt_interval and capture_mode)
Trigger YOLOv8s ONNX inference with one click, and view predictions in both list and histogam form.
All changes are immediately reflected in the Raspberry Pi capture behavior via Nextcloud sync.

* Built using `app.py`, `server.py`, and `ui.py`
* Displays:
  * Predicted images from Nextcloud
  * Filter by leaf class
  * Live camera configuration parameters from `config.json`
  * Time series graph(currently working)
    
* Allows:
  * Refreshing predictions with a button
  * Updating Raspberry Pi camera settings (resolution, measurement interval)
![Shiny_app image7](https://github.com/user-attachments/assets/1f55b570-7be5-48e9-ba8f-3619b3525dbb)


---

## 🐳 Deployment

* The full application is **containerized using Docker**
* A **GitLab CI/CD pipeline** automates build and deployment
* Application is **deployed on Kubernetes**, providing scalable access to the team

---

## 🗂 Folder Structure (on Nextcloud)

```
remote_sensing_data/
├── RSPi_leaf_images/               # Input images from Raspberry Pi
├── Predicted_Images/              # Output from YOLO predictions
├── Yolov8s_Object_Detection_1.onnx  # ONNX model file
├── config.json                    # Camera configuration
├── last_processed.txt          # Timestamp tracker for prediction
```

---

## 👨‍💻 Author

* Developed by: \Mani Varun Arivini — Master’s Thesis Project
