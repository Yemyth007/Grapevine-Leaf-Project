# UAV-Based Grape Leaf Disease Detection System

This repository contains the core implementation of my undergraduate final year project on UAV-based grape leaf disease detection using computer vision and deep learning. The project focuses on building a practical visual perception pipeline capable of performing real-time disease inference from grape leaf images captured using camera-based platforms and deployed on lightweight edge hardware.

Rather than focusing only on offline model training, the work emphasizes end-to-end system integration, including model deployment, real-time inference, and user-facing visualization.

The system pipeline includes image acquisition from UAV and ground-level camera testing, image preprocessing, deep learning-based disease detection, ONNX model inference optimization, Raspberry Pi deployment support, and a web-based interface for result visualization. The overall goal was to demonstrate how AI-driven computer vision can be applied in real agricultural monitoring scenarios under practical constraints.

## Repository Structure

The repository is organized as follows:

- `gui/` – Graphical user interface components used for system interaction  
- `training_results/` – Model training outputs and evaluation artifacts  
- `web_interface/` – Web-based visualization interface used during project demonstrations  
- `rpi5.py` – Raspberry Pi inference and deployment script  
- `best.onnx` – Trained deep learning model exported in ONNX format  
- `requirements.txt` – Required Python packages and dependencies  
- `autorun_website.sh` – Script used to automate web interface startup during deployment  
- `lxterminal.desktop` – Desktop shortcut used during system testing  

## Deployment and Testing Notes

During the original project implementation, the web interface was deployed using Firebase hosting to support live demonstrations and result visualization. The current repository preserves the core inference pipeline, deployment scripts, and system components, while active web hosting is not maintained.

The Raspberry Pi deployment script (`RPI5.py`) was used to handle real-time inference on edge hardware, allowing the system to operate without reliance on high-end computing resources.

Testing was conducted using UAV-captured and ground-level grape leaf images under semi-controlled outdoor conditions to evaluate detection performance and system stability.

## Tools and Technologies

The project was developed using the following tools and technologies:

- Python  
- OpenCV  
- ONNX Runtime  
- Deep learning frameworks for training and model conversion  
- Raspberry Pi for edge deployment  
- Web-based visualization tools  

## Limitations and Future Improvements

This project was developed under undergraduate research constraints and therefore has several limitations. These include limited dataset size due to real-world data collection challenges, testing performed under semi-controlled environmental conditions, and the absence of full autonomous UAV navigation integration.

Future improvements could focus on expanding the dataset, improving model robustness under varying lighting and environmental conditions, optimizing inference performance for low-power edge devices, integrating privacy-aware data handling mechanisms, and extending the system toward autonomous UAV-based crop monitoring.

## Author

Segun Folawewo  
Bachelor of Science in Computer Engineering  
University of the Cordilleras
