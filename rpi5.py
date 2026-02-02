import cv2
from ultralytics import RTDETR
import time
import numpy as np
import os
from picamera2 import Picamera2
import pyrebase
from datetime import datetime
import io

# GUI Imports (PyQt5)
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

class VideoThreadPiCam(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):  
        super().__init__(parent)
        self.parent = parent          
        self._run_flag = True
        self.model = RTDETR('/home/grapeleaf/Downloads/best.onnx')
        self.conf_threshold = 0.8
        self.start_time = time.time()
        self.frame_count = 0
        self.object_counts = {}

        # Firebase configuration
        # Grapevine Leaf Firebase Node Project
        config = {  
            "apiKey": "",
            "authDomain": "",
            "databaseURL": "",
            "projectId": "",
            "storageBucket": "",
            "messagingSenderId": "",
            "appId": "",
            "measurementId": ""
        }

        firebase = pyrebase.initialize_app(config)
        self.storage = firebase.storage()  # Get a reference to Firebase Storage
        self.database = firebase.database()
        self.database.child('total_counts').remove()
        self.database.child('detections').remove()

    def run(self):
        picam2 = Picamera2()
        picam2.start_preview()
        camera_config = picam2.create_video_configuration(
            main={"size": (640, 640), "format": "RGB888"},
            raw={"size": (640, 640)}
        )
        picam2.configure(camera_config)
        picam2.start()

        prev_frame_time = 0
        detected_objects = set()
        disease_counts = {}  

        while self._run_flag:
            frame = picam2.capture_array()
            results = self.parent.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()

            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(annotated_frame, f"{fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            self.object_counts = {}
            for box in results[0].boxes:
                try:
                    class_id = box.cls[0].item()
                    confidence = box.conf.item()
                    label = self.parent.model.names[class_id] if 0 <= class_id < len(self.parent.model.names) else "Unknown"

                    if confidence >= self.conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        # Filter before processing
                        if label != "Not-Grapevine-Leaf":  
                            cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                            self.object_counts[label] = self.object_counts.get(label, 0) + 1
                            current_date = datetime.now().strftime("%Y-%m-%d")

                            for lbl in (label if isinstance(label, list) else [label]):
                                safe_label = lbl.replace(".", "_").replace("#", "_").replace("$", "_").replace("[", "_").replace("]", "_")
                                label_path = f"Detections/{current_date}/{safe_label}"

                                new_node_key = str(int(time.time()))

                                # Screenshot code (upload to Firebase Storage)
                                screenshot_filename = f"{safe_label}_{time.ctime()}.jpg"
                                cloud_path = f"Detections/{current_date}/{screenshot_filename}"

                                # Encode the image data in memory
                                is_success, buffer = cv2.imencode(".jpg", annotated_frame)
                                if not is_success:
                                    print("Error encoding image for upload.")
                                    continue

                                # Upload to Firebase Storage using BytesIO
                                image_stream = io.BytesIO(buffer)
                                self.storage.child(cloud_path).put(image_stream)
                                
                                # Get the download URL of the uploaded image
                                image_url = self.storage.child(cloud_path).get_url(None)

                                detection_data = {
                                    'time': time.ctime(),
                                    'image_url': image_url
                                }

                                self.database.child(label_path).child(new_node_key).set(detection_data)
                                detected_objects.add(label)

                except Exception as e:
                    print(f"Error in detection loop at line {e.__traceback__.tb_lineno}: {e}")

            detected_objects.clear()
            self.change_pixmap_signal.emit(annotated_frame)
            self.send_to_firebase(self.object_counts)


            
    def send_to_firebase(self, object_counts):
        total_counts = self.database.child('total_counts').get().val()  # Get existing total counts
        if total_counts is None:  # Initialize if it doesn't exist
            total_counts = {}

        # Update total counts
        for label, count in object_counts.items():
            total_counts[label] = total_counts.get(label, 0) + count
        
        # Write the updated total counts to Firebase
        self.database.child('total_counts').set(total_counts)

    def stop(self):
        self._run_flag = False
        self.wait()
        
class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grapevine Leaf Object Detection")

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)  # Fixed size for display

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        self.setLayout(vbox)

        # Load the PyTorch model in the main thread
        self.model = RTDETR('/home/grapeleaf/Downloads/latestModel.onnx') 

        self.thread = VideoThreadPiCam(self)  # Pass 'self' to the thread
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        qt_img = self.convert_cv_qt(frame)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        """Convert OpenCV image to QPixmap"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio) 
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    # Fix for Threading issue
    app = QApplication([])
    app.setAttribute(QtCore.Qt.AA_X11InitThreads)  # Initialize for X11 threads
    window = App()
    window.show()
    app.exec_()

    # Create screenshots directory in the home directory
    os.makedirs("/home/grapeleaf/Downloads/screenshots", exist_ok=True)  
