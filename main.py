import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta

# Path to the training images
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print("Image List:", myList)

# Load images and extract class names (full names)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Class Names:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Dictionary to store the last attendance time for each person
last_attendance_time = {}

def markAttendance(name):
    now = datetime.now()
    dayString = now.strftime('%A')  # Day of the week
    dtString = now.strftime('%H:%M:%S')  # Time

    # Check if the person has been marked within the last minute
    if name in last_attendance_time:
        if now - last_attendance_time[name] < timedelta(minutes=1):
            return  # Skip if attendance was marked within the last minute

    # Update the last attendance time
    last_attendance_time[name] = now

    # Check and add header if not present
    file_path = 'attendance.csv'
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as f:
            f.write('Name,Day,Time\n')
    
    else:
        with open(file_path, 'r+') as f:
            first_line = f.readline().strip()
            if first_line != 'Name,Day,Time':
                f.seek(0, 0)
                lines = f.readlines()
                f.seek(0)
                f.write('Name,Day,Time\n')
                for line in lines:
                    f.write(line)
    
    # Write the attendance data to the file
    with open(file_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            f.writelines(f'\n{name},{dayString},{dtString}')
        else:
            f.writelines(f'\n{name},{dayString},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
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

            # Calculate text size to adjust the box size
            (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x1 + text_width + 12, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    
    # Press 'x' to exit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
