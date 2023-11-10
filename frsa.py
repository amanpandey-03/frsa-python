import os
import cv2
import face_recognition
import csv
from datetime import datetime, timedelta

# Directory where your photos are stored
photos_directory = "photos/"

# Get the list of image files in the directory
image_files = [f for f in os.listdir(photos_directory) if f.endswith((".jpg", ".jpeg", ".png"))]

# List to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Populate known face encodings and names from the images in the directory
for image_file in image_files:
    image_path = os.path.join(photos_directory, image_file)
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(image_file)[0])


video_capture = cv2.VideoCapture(0)


current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"{current_date}_attendance.csv"


def recognize_faces_in_frame(frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if not face_encodings:
        return "Unknown"

    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
    
    name = "Unknown"
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    return name


# Open CSV file for appending
with open(csv_filename, mode="a", newline="") as csv_file:
    fieldnames = ["Name", "Date", "Time"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header to CSV file if the file is empty
    if os.stat(csv_filename).st_size == 0:
        writer.writeheader()

    
    attendance_recorded = {name: False for name in known_face_names}

    
    end_time = datetime.now() + timedelta(seconds=3)

    while datetime.now() < end_time:

        ret, frame = video_capture.read()

        
        name = recognize_faces_in_frame(frame, known_face_encodings, known_face_names)

        
        if name != "Unknown" and not attendance_recorded[name]:
            
            writer.writerow({"Name": name, "Date": current_date, "Time": datetime.now().strftime("%H:%M:%S")})
            attendance_recorded[name] = True

        
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


video_capture.release()
cv2.destroyAllWindows()

print(f"Attendance recorded in {csv_filename}")
