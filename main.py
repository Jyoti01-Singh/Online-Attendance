import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime, timedelta

# Define paths and initialize variables
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load images and names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    file_path = 'D://python//online_attenadance_system//attendance.csv'
    
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dayString = now.strftime('%A')  # Get the day of the week

    # Write data to the CSV file
    try:
        # Check if the file exists to determine if the header is needed
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # Write header if file does not exist
            if not file_exists:
                writer.writerow(['Name', 'Time', 'Day'])
            # Write the attendance data
            writer.writerow([name, dtString, dayString])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

# Dictionary to keep track of the last attendance time for each student
last_attendance_time = {}

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success or img is None or img.size == 0:
        print("Failed to capture image or empty frame")
        continue

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Get the current time
            now = datetime.now()

            # Check if this student's last recorded time exists
            if name in last_attendance_time:
                # Check if a minute has passed since the last attendance
                last_time = last_attendance_time[name]
                if now - last_time >= timedelta(minutes=1):
                    markAttendance(name)
                    last_attendance_time[name] = now
            else:
                # Record attendance for the first time
                markAttendance(name)
                last_attendance_time[name] = now

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
