# face_recognition_app.py
import cv2
import face_recognition
import pickle

def extract_features(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    return face_recognition.face_encodings(rgb_frame, face_locations), face_locations

def recognize_faces(frame, model):
    face_encodings, face_locations = extract_features(frame)
    predictions = [model.predict([encoding])[0] for encoding in face_encodings]
    return predictions, face_locations

if __name__ == "__main__":
    model_path = 'model/face_recognition_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions, face_locations = recognize_faces(frame, model)
        for (top, right, bottom, left), name in zip(face_locations, predictions):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
