🌊 Aquadex AI

AI-Based Underwater Sonar Analysis and Shipwreck Detection

Aquadex AI is a hackathon prototype designed to analyze underwater/side-scan sonar images and identify possible shipwreck anomalies using an Ultralytics YOLO11 segmentation model.

The system also provides GPS information, map visualization, annotated detection results, and downloadable reports.

🚀 Project Features

Upload an underwater or sonar image

YOLO11-based shipwreck detection and segmentation

Red detection visualization

Confidence-based detection

GPS latitude and longitude information

GPS metadata processing

Map visualization

Detection result reporting

Annotated output image

JSON/report generation

Streamlit-based user interface

🤖 AI Model

Model: YOLO11n-Seg

Task: Image segmentation

Current trained class: Shipwreck

Model file: model/best.pt

The current trained model is specifically for shipwreck detection. It is not a multi-class marine-debris classifier.

📁 Project Structure

Aquadex-AI/
│
├── app/
│   └── app.py
│
├── demo/
│   ├── Sonar image 1.png
│   ├── Sonar image 2.png
│   ├── Sonar image 3.png
│   └── Sonar image 4.png
│
├── gps/
│   └── metadata_parser.py
│
├── inference/
│   └── predict.py
│
├── model/
│   └── best.pt
│
├── reports/
│
├── requirements.txt
│
└── README.md

🛠️ Technologies Used

Python

Streamlit

Ultralytics YOLO11

YOLO11n-Seg

OpenCV

NumPy

Pandas

Pillow

GPS metadata processing

🔄 System Workflow

Underwater / Sonar Image
          ↓
     Image Upload
          ↓
       YOLO11
          ↓
 Debris Detection / Segmentation
          ↓
   Detection Visualization
          ↓
    GPS Information
          ↓
     Map & Report

📍 GPS Module

The GPS module reads latitude and longitude information from metadata.

File:

gps/metadata_parser.py

The GPS information can be used to associate an analyzed image with a geographic location.

🔎 Inference Module

The inference code is located at:

inference/predict.py

It loads:

model/best.pt

and performs YOLO inference on the input image.

🖥️ Running the Application

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app/app.py

🎯 Hackathon Purpose

Aquadex AI demonstrates how computer vision can be applied to underwater sonar imagery to assist in identifying shipwreck anomalies and connecting detections with geographic information.

This project is a hackathon prototype intended for demonstration and further development.

⚠️ Current Limitation

The current YOLO11 model has one trained class:

shipwreck

Therefore, the current AI model should not be interpreted as a trained detector for plastic, pipes, fishing nets, metal debris, or other individual debris categories.

Future versions can expand the model with additional labeled underwater/sonar classes.

👨‍💻 Project

Aquadex AI

AI-powered underwater sonar analysis and shipwreck detection prototype.🌊 Aquadex AI

AI-Based Underwater Sonar Analysis and Shipwreck Detection

Aquadex AI is a hackathon prototype designed to analyze underwater/side-scan sonar images and identify possible shipwreck anomalies using an Ultralytics YOLO11 segmentation model.

The system also provides GPS information, map visualization, annotated detection results, and downloadable reports.

🚀 Project Features

Upload an underwater or sonar image

YOLO11-based shipwreck detection and segmentation

Red detection visualization

Confidence-based detection

GPS latitude and longitude information

GPS metadata processing

Map visualization

Detection result reporting

Annotated output image

JSON/report generation

Streamlit-based user interface

🤖 AI Model

Model: YOLO11n-Seg

Task: Image segmentation

Current trained class: Shipwreck

Model file: model/best.pt

The current trained model is specifically for shipwreck detection. It is not a multi-class marine-debris classifier.

📁 Project Structure

Aquadex-AI/
│
├── app/
│   └── app.py
│
├── demo/
│   ├── Sonar image 1.png
│   ├── Sonar image 2.png
│   ├── Sonar image 3.png
│   └── Sonar image 4.png
│
├── gps/
│   └── metadata_parser.py
│
├── inference/
│   └── predict.py
│
├── model/
│   └── best.pt
│
├── reports/
│
├── requirements.txt
│
└── README.md

🛠️ Technologies Used

Python

Streamlit

Ultralytics YOLO11

YOLO11n-Seg

OpenCV

NumPy

Pandas

Pillow

GPS metadata processing

🔄 System Workflow

Underwater / Sonar Image
          ↓
     Image Upload
          ↓
       YOLO11
          ↓
Shipwreck Detection / Segmentation
          ↓
   Detection Visualization
          ↓
    GPS Information
          ↓
     Map & Report

📍 GPS Module

The GPS module reads latitude and longitude information from metadata.

File:

gps/metadata_parser.py

The GPS information can be used to associate an analyzed image with a geographic location.

🔎 Inference Module

The inference code is located at:

inference/predict.py

It loads:

model/best.pt

and performs YOLO inference on the input image.

🖥️ Running the Application

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app/app.py

🎯 Hackathon Purpose

Aquadex AI demonstrates how computer vision can be applied to underwater sonar imagery to assist in identifying shipwreck anomalies and connecting detections with geographic information.

This project is a hackathon prototype intended for demonstration and further development.

⚠️ Current Limitation

The current YOLO11 model has one trained class:

shipwreck

Therefore, the current AI model should not be interpreted as a trained detector for plastic, pipes, fishing nets, metal debris, or other individual debris categories.

Future versions can expand the model with additional labeled underwater/sonar classes.

👨‍💻 Project

Aquadex AI

AI-powered underwater sonar analysis and shipwreck detection prototype.🌊 Aquadex AI

AI-Based Underwater Sonar Analysis and Shipwreck Detection

Aquadex AI is a hackathon prototype designed to analyze underwater/side-scan sonar images and identify possible shipwreck anomalies using an Ultralytics YOLO11 segmentation model.

The system also provides GPS information, map visualization, annotated detection results, and downloadable reports.

🚀 Project Features

Upload an underwater or sonar image

YOLO11-based shipwreck detection and segmentation

Red detection visualization

Confidence-based detection

GPS latitude and longitude information

GPS metadata processing

Map visualization

Detection result reporting

Annotated output image

JSON/report generation

Streamlit-based user interface

🤖 AI Model

Model: YOLO11n-Seg

Task: Image segmentation

Current trained class: Shipwreck

Model file: model/best.pt

The current trained model is specifically for shipwreck detection. It is not a multi-class marine-debris classifier.

📁 Project Structure

Aquadex-AI/
│
├── app/
│   └── app.py
│
├── demo/
│   ├── Sonar image 1.png
│   ├── Sonar image 2.png
│   ├── Sonar image 3.png
│   └── Sonar image 4.png
│
├── gps/
│   └── metadata_parser.py
│
├── inference/
│   └── predict.py
│
├── model/
│   └── best.pt
│
├── reports/
│
├── requirements.txt
│
└── README.md

🛠️ Technologies Used

Python

Streamlit

Ultralytics YOLO11

YOLO11n-Seg

OpenCV

NumPy

Pandas

Pillow

GPS metadata processing

🔄 System Workflow

Underwater / Sonar Image
          ↓
     Image Upload
          ↓
       YOLO11
          ↓
Shipwreck Detection / Segmentation
          ↓
   Detection Visualization
          ↓
    GPS Information
          ↓
     Map & Report

📍 GPS Module

The GPS module reads latitude and longitude information from metadata.

File:

gps/metadata_parser.py

The GPS information can be used to associate an analyzed image with a geographic location.

🔎 Inference Module

The inference code is located at:

inference/predict.py

It loads:

model/best.pt

and performs YOLO inference on the input image.

🖥️ Running the Application

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app/app.py

🎯 Hackathon Purpose

Aquadex AI demonstrates how computer vision can be applied to underwater sonar imagery to assist in identifying shipwreck anomalies and connecting detections with geographic information.

This project is a hackathon prototype intended for demonstration and further development.

⚠️ Current Limitation

The current YOLO11 model has one trained class:

shipwreck

Therefore, the current AI model should not be interpreted as a trained detector for plastic, pipes, fishing nets, metal debris, or other individual debris categories.

Future versions can expand the model with additional labeled underwater/sonar classes.

👨‍💻 Project

Aquadex AI

AI-powered underwater sonar analysis and Debris detection prototype.