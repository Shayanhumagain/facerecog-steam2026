import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


label_map = {}
dataset_path = "dataset"
for i, name in enumerate(os.listdir(dataset_path)):
    label_map[i] = name

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1536)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 864)


cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Face Recognition",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)

        if confidence < 52.7:
            name = label_map[label]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({int(confidence)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,255,255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
