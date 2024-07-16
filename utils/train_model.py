# train_model.py
import os
import pickle
from sklearn import svm
import face_recognition
import cv2

def extract_features(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    return face_recognition.face_encodings(rgb_image, face_locations)

def prepare_dataset(dataset_path):
    features, labels = [], []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            face_encodings = extract_features(image_path)
            if face_encodings:
                features.append(face_encodings[0])
                labels.append(person_name)
    return features, labels

def train_model(features, labels, model_save_path):
    if len(set(labels)) <= 1:
        raise ValueError("The number of classes has to be greater than one; got 1 class")

    clf = svm.SVC(gamma='scale')
    clf.fit(features, labels)
    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model trained and saved to {model_save_path}")

if __name__ == "__main__":
    dataset_path = '../dataset'
    model_path = '../model'
    model_save_path = os.path.join(model_path, "face_recognition_model.pkl")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    features, labels = prepare_dataset(dataset_path)
    train_model(features, labels, model_save_path)
