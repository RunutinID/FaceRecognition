# collect_dataset.py
import cv2
import os

def capture_images(user_name, save_path, num_images=10):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)
    count = 0
    print("Press SPACE to capture images. Press ESC to exit.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Images", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            img_name = os.path.join(save_path, f"{user_name}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name}")
            count += 1
        elif key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    dataset_path = os.path.join("../dataset", user_name)
    capture_images(user_name, dataset_path)
    print(f"Dataset for {user_name} is ready at {dataset_path}")
