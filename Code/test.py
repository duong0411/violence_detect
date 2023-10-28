import numpy as np
import cv2
from keras.models import load_model
from collections import deque
import time

def print_results():
    print("Loading model ...")
    model = load_model("C:/Users/Admin/PycharmProjects/pythonProject5/Code/modelnew3.h5")
    Q = deque(maxlen=128)

    vs = cv2.VideoCapture(0)  # 0 corresponds to the default camera (webcam)

    writer = None
    prev_time = time.time()

    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float64")
        frame = frame.reshape(128, 128, 3) / 255

        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        results = np.array(Q).mean(axis=0)
        i = (preds > 0.7)[0]
        label = i
        if label:  # Violence prob
            text_color = (0, 0, 255)  # red
        else:
            text_color = (0, 255, 0)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output/v_output.avi", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        writer.write(output)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(output, fps_text, (10, 30), FONT, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Webcam", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    print("[INFO] Cleaning up...")
    writer.release()
    vs.release()
    cv2.destroyAllWindows()

print_results()
