from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml


# Function to predict and save images
def predict_and_save(model, image_path, output_path, output_path_txt):
    # Perform prediction
    results = model.predict(image_path, conf=0.5)

    result = results[0]
    # Draw boxes on the image
    img = result.plot()  # Plots the predictions directly on the image

    # Save the result
    cv2.imwrite(str(output_path), img)
    # Save the bounding box data
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywh[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':

    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            # yolo_params.yaml test field already points to the images directory
            images_dir = Path(data['test'])
        else:
            print("No test field found in yolo_params.yaml, please add the test field with the path to the test images")
            raise SystemExit()

    # check that the images directory exists
    if not images_dir.exists():
        print(f"Images directory {images_dir} does not exist")
        raise SystemExit()

    if not images_dir.is_dir():
        print(f"Images directory {images_dir} is not a directory")
        raise SystemExit()

    if not any(images_dir.iterdir()):
        print(f"Images directory {images_dir} is empty")
        raise SystemExit()

    # Load the YOLO model
    detect_path = this_dir / "runs" / "detect"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path / f) and f.startswith("train")]
    if len(train_folders) == 0:
        raise ValueError("No training folders found")
    # Automatically select the most recently modified training folder
    train_folders.sort(key=lambda f: os.path.getmtime(detect_path / f))
    idx = len(train_folders) - 1
    print(f"Using training folder: {train_folders[idx]}")

    model_path = detect_path / train_folders[idx] / "weights" / "best.pt"
    model = YOLO(model_path)

    # Directory with images
    output_dir = this_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create images and labels subdirectories
    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through the images in the directory
    for img_path in images_dir.glob('*'):
        if img_path.suffix not in ['.png', '.jpg']:
            continue
        output_path_img = images_output_dir / img_path.name
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, output_path_img, output_path_txt)

    print(f"Predicted images saved in {images_output_dir}")
    print(f"Bounding box labels saved in {labels_output_dir}")
    data = this_dir / 'yolo_params.yaml'
    print(f"Model parameters saved in {data}")
    metrics = model.val(data=data, split="test")
