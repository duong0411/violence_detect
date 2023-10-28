import sys
import cv2
import onnxruntime
import numpy as np
from collections import deque
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QGridLayout, QWidget
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from PyQt5.QtCore import QThread, pyqtSignal

class EmailSender(QThread):
    email_sent = pyqtSignal(bool)
    def __init__(self, sender_email, sender_password, receiver_email, frame):
        super().__init__()
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
        self.frame = frame

    def run(self):
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        msg["Subject"] = "Violence Detection"
        message = "Violence detected in the following image."
        msg.attach(MIMEText(message, "plain"))
        image_path = "temp_image.jpg"
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, self.frame)
        with open(image_path, "rb") as img_file:
            image = MIMEImage(img_file.read())
            msg.attach(image)
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
            server.quit()
            self.email_sent.emit(True)
        except Exception as e:
            print("Error sending email:", str(e))
            self.email_sent.emit(False)

class RTSPStreamer(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_processed.emit(frame)
        cap.release()

class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_path = "model.onnx"
        self.confidence_threshold = 0.7
        self.input_size = (128, 128)
        self.session = onnxruntime.InferenceSession(self.model_path)
        self.Q = deque(maxlen=128)
        self.vs1 = None
        self.vs2 = None
        self.initUI()
        self.sender_email = "duong04112000@gmail.com"
        self.sender_password = "pqfo bpvy zvfm abpk"
        self.receiver_email = "duog0411@gmail.com"
        self.setWindowTitle("Violence Detection App")
        self.setGeometry(800, 600, 600, 800)

        self.email_sender1 = EmailSender(self.sender_email, self.sender_password, self.receiver_email, None)
        self.email_sender1.email_sent.connect(self.email_sent_handler1)

        self.email_sender2 = EmailSender(self.sender_email, self.sender_password, self.receiver_email, None)
        self.email_sender2.email_sent.connect(self.email_sent_handler2)

    def initUI(self):
        self.video_label1 = QLabel(self)
        self.video_label2 = QLabel(self)
        self.start_button1 = QPushButton("Start Camera 1", self)
        self.start_button2 = QPushButton("Start Camera 2", self)
        self.ip_input1 = QLineEdit(self)
        self.ip_input2 = QLineEdit(self)

        layout = QGridLayout()
        layout.addWidget(self.ip_input1, 0, 0)
        layout.addWidget(self.start_button1, 0, 1)
        layout.addWidget(self.video_label1, 1, 0)
        layout.addWidget(self.ip_input2, 2, 0)
        layout.addWidget(self.start_button2, 2, 1)
        layout.addWidget(self.video_label2, 3, 0)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.update_frame1)
        self.timer1.start(1000 // 30)  # Update frame at 30 fps

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.update_frame2)
        self.timer2.start(1000 // 30)  # Update frame at 30 fps

        self.start_button1.clicked.connect(self.start_stream1)
        self.start_button2.clicked.connect(self.start_stream2)

        self.start_button1.setGeometry(350, 20, 150, 30)
        self.start_button2.setGeometry(350, 250, 150, 30)
        self.ip_input1.setGeometry(350, 20, 150, 30)


    def start_stream1(self):
        ip_address = self.ip_input1.text()
        rtsp_url = f"rtsp://duong0411:duong04112000@{ip_address}:554/stream1"
        self.vs1 = RTSPStreamer(rtsp_url)
        self.vs1.frame_processed.connect(self.process_frame1)
        self.vs1.start()

    def start_stream2(self):
        ip_address = self.ip_input2.text()
        rtsp_url = f"rtsp://duong0411:duong04112000@{ip_address}:554/stream1"
        self.vs2 = RTSPStreamer(rtsp_url)
        self.vs2.frame_processed.connect(self.process_frame2)
        self.vs2.start()

    def process_frame1(self, frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(frame, self.input_size).astype("float32") / 255
        input_data = np.expand_dims(input_data, axis=0)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_data})
        preds = result[0][0]
        self.Q.append(preds)
        label = (preds > self.confidence_threshold)[0]
        if label:
            text_color = (0, 0, 255)
            if not self.email_sender1.isRunning():
                self.email_sender1.frame = frame
                self.email_sender1.start()
        else:
            text_color = (0, 255, 0)
        text = "Violence: {}".format(label)
        cv2.putText(frame, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 3)
        self.display_frame(self.video_label1, frame)

    def process_frame2(self, frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(frame, self.input_size).astype("float32") / 255
        input_data = np.expand_dims(input_data, axis=0)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_data})
        preds = result[0][0]
        self.Q.append(preds)
        label = (preds > self.confidence_threshold)[0]
        if label:
            text_color = (0, 0, 255)
            if not self.email_sender2.isRunning():
                self.email_sender2.frame = frame
                self.email_sender2.start()
        else:
            text_color = (0, 255, 0)
        text = "Violence: {}".format(label)
        cv2.putText(frame, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 3)
        self.display_frame(self.video_label2, frame)

    def update_frame1(self):
        pass

    def update_frame2(self):
        pass

    def display_frame(self, label, frame):
        h, w, ch = frame.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(600, 600, Qt.KeepAspectRatio)
        label.setPixmap(QPixmap.fromImage(p))

    def email_sent_handler1(self, success):
        if success:
            print("Email 1 sent successfully.")
        else:
            print("Failed to send email 1.")

    def email_sent_handler2(self, success):
        if success:
            print("Email 2 sent successfully.")
        else:
            print("Failed to send email 2.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())

