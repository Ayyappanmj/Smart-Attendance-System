import os
import threading
import configparser
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import cv2
    if not hasattr(cv2, 'face'):
        raise ImportError('opencv-contrib-python')
    import numpy as np
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import Label, Button, Text, Scrollbar, VERTICAL, RIGHT, Y, END
    from tkinter import simpledialog
except (ModuleNotFoundError, ImportError) as e:
    missing_package = str(e).split("'")[1] if "'" in str(e) else str(e)
    raise SystemExit(
        f"Missing dependency: {missing_package}.\n"
        "Install the required packages with:\n"
        "    python -m pip install -r requirements.txt\n"
        "Then run the script again.\n"
        "If the missing dependency is opencv-contrib-python, install it instead of opencv-python."
    )

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
label_to_name = {}
name_to_label = {}
is_face_recognizer_trained = False

# Load configuration settings
def load_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(BASE_DIR, 'config.ini')
    config.read(config_path)
    global KNOWN_FACES_DIR, ATTENDANCE_CSV_PATH, CAMERA_INDEX, PASSWORD
    known_faces_dir = config.get('Paths', 'KNOWN_FACES_DIR', fallback='known_faces')
    attendance_path = config.get('Paths', 'ATTENDANCE_CSV_PATH', fallback='attendance.csv')
    KNOWN_FACES_DIR = os.path.join(BASE_DIR, known_faces_dir) if not os.path.isabs(known_faces_dir) else known_faces_dir
    ATTENDANCE_CSV_PATH = os.path.join(BASE_DIR, attendance_path) if not os.path.isabs(attendance_path) else attendance_path
    CAMERA_INDEX = config.getint('Camera', 'CAMERA_INDEX', fallback=0)
    PASSWORD = config.get('Security', 'PASSWORD', fallback='admin')

load_config()

# List to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Set to track attendance in the current session
attendance_set = set()

# Load known faces and train the OpenCV LBPH recognizer
def load_known_faces():
    global label_to_name, name_to_label, is_face_recognizer_trained
    label_to_name.clear()
    name_to_label.clear()
    is_face_recognizer_trained = False
    training_images = []
    training_labels = []
    next_label = 0

    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        update_status(f"Known faces directory created: {KNOWN_FACES_DIR}", "orange")
        return

    try:
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) == 0:
                    continue

                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                if face.size == 0:
                    continue

                if name not in name_to_label:
                    name_to_label[name] = next_label
                    label_to_name[next_label] = name
                    next_label += 1

                training_images.append(face)
                training_labels.append(name_to_label[name])

        if training_images:
            face_recognizer.train(training_images, np.array(training_labels))
            is_face_recognizer_trained = True
            update_status("Faces loaded successfully.", "green")
        else:
            update_status("No known faces found. Add images to the known faces directory.", "orange")
    except Exception as e:
        update_status(f"Error loading faces: {e}", "red")

# Record attendance in a CSV file if not already recorded in the current session
def mark_attendance(name):
    if name not in attendance_set:
        try:
            target_dir = os.path.dirname(ATTENDANCE_CSV_PATH)
            if target_dir and not os.path.isdir(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            if not os.path.exists(ATTENDANCE_CSV_PATH):
                with open(ATTENDANCE_CSV_PATH, 'w') as f:
                    f.write('Name,DateTime\n')  # Header for the CSV file

            with open(ATTENDANCE_CSV_PATH, 'a') as f:
                now = datetime.now()
                dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'{name},{dt_string}\n')
            attendance_set.add(name)
            attendance_log.insert(END, f'{name} marked present at {dt_string}\n')
            attendance_log.see(END)
            update_status(f"Attendance marked for {name}.", "green")
        except Exception as e:
            update_status(f"Error marking attendance: {e}", "red")

# Start the camera and process frames
def find_working_camera(preferred_index=0, max_index=4):
    for index in range(preferred_index, max_index):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            capture.release()
            return index
        capture.release()
    return None


def update_frame(imgtk):
    display_frame.imgtk = imgtk
    display_frame.configure(image=imgtk)


def start_camera(camera_index=0):
    load_known_faces()
    selected_index = camera_index
    capture = cv2.VideoCapture(selected_index)
    if not capture.isOpened():
        capture.release()
        fallback_index = find_working_camera(0, 4)
        if fallback_index is not None:
            selected_index = fallback_index
            update_status(f"Camera {camera_index} unavailable. Using camera {selected_index}.", "orange")
        else:
            update_status("Unable to open any camera. Check your camera connection and CAMERA_INDEX.", "red")
            return

    video_capture = cv2.VideoCapture(selected_index)
    if not video_capture.isOpened():
        update_status("Unable to open camera. Check CAMERA_INDEX.", "red")
        return

    while camera_running:
        ret, frame = video_capture.read()
        if not ret:
            update_status("Failed to capture video.", "red")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            name = "Unknown"
            face_image = gray_frame[y:y + h, x:x + w]

            if is_face_recognizer_trained and face_image.size > 0:
                try:
                    label, confidence = face_recognizer.predict(face_image)
                    if confidence < 70:
                        name = label_to_name.get(label, "Unknown")
                except Exception:
                    name = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if name != "Unknown":
                mark_attendance(name)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        root.after(0, update_frame, imgtk)

    video_capture.release()
    cv2.destroyAllWindows()

# Stop the camera
def stop_camera():
    global camera_running
    camera_running = False
    update_status("Camera stopped.", "blue")

# Start a new session
def start_new_session():
    global attendance_set
    attendance_set.clear()
    attendance_log.delete(1.0, END)
    update_status("New session started.", "green")

# Update status in the GUI
def update_status(message, color="black"):
    status_label.config(text=f"Status: {message}", fg=color)

# Authenticate user before starting the application
def authenticate_user():
    auth_root = tk.Tk()
    auth_root.withdraw()
    user_input = simpledialog.askstring("Authentication", "Enter password:", show='*', parent=auth_root)
    auth_root.destroy()

    if user_input != PASSWORD:
        print("Access denied")
        exit(1)
    print("Access granted")

# Main function to initialize the GUI
def main():
    authenticate_user()

    global root, display_frame, attendance_log, status_label, camera_running
    root = tk.Tk()
    root.title("Smart Attendance System")

    display_frame = Label(root)
    display_frame.pack()

    start_button = Button(root, text="Start Camera", command=lambda: threading.Thread(target=start_camera_thread, daemon=True).start())
    start_button.pack()

    stop_button = Button(root, text="Stop Camera", command=stop_camera)
    stop_button.pack()

    new_session_button = Button(root, text="New Session", command=start_new_session)
    new_session_button.pack()

    status_label = Label(root, text="Status: Waiting for action", fg="blue")
    status_label.pack()

    scrollbar = Scrollbar(root, orient=VERTICAL)

    attendance_log = Text(root, height=10, yscrollcommand=scrollbar.set)
    attendance_log.pack(side=RIGHT)

    scrollbar.config(command=attendance_log.yview)
    scrollbar.pack(side=RIGHT, fill=Y)

    camera_running = False

    root.mainloop()

# Start camera thread
def start_camera_thread():
    global camera_running
    camera_running = True
    start_camera(CAMERA_INDEX)

if __name__ == "__main__":
    main()
