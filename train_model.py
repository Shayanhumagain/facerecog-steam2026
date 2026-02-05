import cv2
import os
import numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

current_label = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

print("Training complete!")
print("Labels:", label_map)
