# Master Thesis: Object_Detection-Python_Shiny_APP deployed on Kubernetes cluster

This project enables automated **leaf object detection** using a YOLOv8s ONNX model and displays the results through a **Shiny web application**. 
The complete system integrates with **Nextcloud** to retrieve and store data, and is designed to work seamlessly with a **Raspberry Pi camera**.

---

## 🔁 Workflow Summary

* A **Raspberry Pi** camera captures unseen leaf images at regular time intervals which is operated from Shiny App.
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

### ✅ Shiny App

* Built using `app.py`, `server.py`, and `ui.py`
* Displays:
  * Predicted images from Nextcloud
  * Filter by leaf class
  * Live camera configuration parameters from `config.json`
  * Time series graph(currently working)
    
* Allows:
  * Refreshing predictions with a button
  * Updating Raspberry Pi camera settings (resolution, measurement interval)
![Shiny_app image3](https://github.com/user-attachments/assets/ae5cfdeb-7c4b-4e03-af22-610ef13e3c86)

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
## ✨ Future Work

* Time-series and class distribution visualizations
* Live confidence tracking
* Image gallery enhancements

---

## 👨‍💻 Author

* Developed by: \Mani Varun Arivini — Master’s Thesis Project
