import threading
import cv2
from deepface import DeepFace
import os
import pandas as pd
from datetime import datetime
from openpyxl import Workbook

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
counter = 0
face_match = False
reference_images = {}
attendance = {}
reference_dir = "D:/testing/reference_imges"
for filename in os.listdir(reference_dir):
    if filename.endswith((".jpg", ".png")):
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(reference_dir, filename)
        reference_images[name] = cv2.imread(img_path)
        attendance[name] = False
def check_face(frame):
    global face_match, attendance
    for name, ref_img in reference_images.items():
        try:
            if DeepFace.verify(frame, ref_img.copy())['verified']:
                face_match = True
                attendance[name] = True
                return
        except ValueError:
            pass
    face_match = False
def update_excel():
    date = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(list(attendance.items()), columns=['Name', 'Present'])
    df['Date'] = date
    df['Present'] = df['Present'].map({True: 'Yes', False: 'No'}) 
    filename = 'attendance.xlsx'
    if os.path.exists(filename):
        existing_df = pd.read_excel(filename)
        updated_df = pd.concat([existing_df, df]).drop_duplicates(subset=['Name', 'Date'], keep='last')
    else:
        updated_df = df
    updated_df.to_excel(filename, index=False)
while True:
    ret, frame = capture.read()
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
update_excel()
cv2.destroyAllWindows()