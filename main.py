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
    if curImg is not None:  # Ensure image is loaded
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])  # Take the first encoding if multiple are found
    return encodeList

def check_and_add_header(file_path):
    header = ['Name', 'Time', 'Day']
    file_exists = os.path.isfile(file_path)

    if file_exists:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            # If the first row is not the expected header, rewrite the file with the correct header
            if existing_header != header:
                print("Header missing or incorrect, adding header...")
                # Read existing data and add header if missing
                data = f.readlines()
                with open(file_path, 'w', encoding='utf-8', newline='') as f_write:
                    writer = csv.writer(f_write)
                    writer.writerow(header)  # Write header
                    for row in data:
                        f_write.write(row)  # Write back existing data
    else:
        # File doesn't exist, so create it and add the header
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def markAttendance(name):
    file_path = 'D://python//online_attenadance_system//attendance.csv'
    
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    dayString = now.strftime('%A')  # Get the day of the week

    # Check and add the header if missing
    check_and_add_header(file_path)
    
    try:
        # Append data to the file
        with open(file_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
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

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS, model='hog')  # Use 'cnn' for higher accuracy
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            now = datetime.now()

            if name in last_attendance_time:
                last_time = last_attendance_time[name]
                if now - last_time >= timedelta(minutes=1):
                    markAttendance(name)
                    last_attendance_time[name] = now
            else:
                markAttendance(name)
                last_attendance_time[name] = now

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
