import os
import json
import ast
import random
import shutil
from PIL import Image

def parse_contour(contour_str):
    return ast.literal_eval(contour_str)

def compute_bbox(contour):
    min_x = min([point[0] for point in contour])
    min_y = min([point[1] for point in contour])
    max_x = max([point[0] for point in contour])
    max_y = max([point[1] for point in contour])
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def compute_area(contour):
    n = len(contour)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += contour[i][0] * contour[j][1]
        area -= contour[j][0] * contour[i][1]
    return abs(area) * 0.5

def generate_coco_annotations(img_id, img_width, img_height, contours):
    annotations = []
    image_info = {
        "id": img_id,
        "width": img_width,
        "height": img_height,
        "file_name": f"image_{img_id}.png"
    }
    ann_id = 1
    for contour_str in contours:
        contour = parse_contour(contour_str)
        if len(contour) < 3:  # Ensure valid polygons
            continue
        bbox = compute_bbox(contour)
        area = compute_area(contour)
        segmentation = [list(sum(contour, ()))]  # Properly flattened list
        annotation = {
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": segmentation,  # Ensuring it's a list of lists
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }
        annotations.append(annotation)
        ann_id += 1
    return image_info, annotations

def create_coco_dataset_from_files(files):
    all_images = []
    all_annotations = []
    img_id = 0
    for filename, txt_filename in files:
        img_id += 1
        img_path = filename
        txt_path = txt_filename
        img = Image.open(img_path)
        img_width, img_height = img.size
        with open(txt_path, 'r') as f:
            contours = [line.strip() for line in f.readlines()]
        image_info, annotations = generate_coco_annotations(img_id, img_width, img_height, contours)
        image_info["file_name"] = os.path.basename(filename)
        all_images.append(image_info)
        all_annotations.extend(annotations)
    return {
        "images": all_images,
        "annotations": all_annotations,
        "categories": [{
            "id": 1,
            "name": "contour"
        }]
    }
if __name__ == "__main__":
    directory = input("Enter the directory path containing the image-text pairs: ")
    train_output_file = input("Enter the desired name for the training output JSON file (e.g., 'train.json'): ")
    val_output_file = input("Enter the desired name for the validation output JSON file (e.g., 'val.json'): ")
    
    # Create train and validation directories to store images
    train_dir = os.path.join(directory, "train_images")
    val_dir = os.path.join(directory, "val_images")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Collect all image-text file pairs
    pairs = [(os.path.join(directory, f), os.path.join(directory, f.replace('.png', '.txt')))
             for f in os.listdir(directory) if f.endswith('.png') and f.replace('.png', '.txt') in os.listdir(directory)]
    
    # Shuffle and split into training and validation sets
    random.shuffle(pairs)
    split_index = int(0.8 * len(pairs))
    train_files = pairs[:split_index]
    val_files = pairs[split_index:]
    
    # Copy train and validation images to their respective directories
    for img_file, _ in train_files:
        shutil.copy(img_file, train_dir)
    for img_file, _ in val_files:
        shutil.copy(img_file, val_dir)
    
    # Create COCO datasets
    train_dataset = create_coco_dataset_from_files(train_files)
    val_dataset = create_coco_dataset_from_files(val_files)
    
    # Save datasets to JSON files
    with open(train_output_file, 'w') as f:
        json.dump(train_dataset, f, indent=4)
    with open(val_output_file, 'w') as f:
        json.dump(val_dataset, f, indent=4)
    
    print(f"Training dataset saved to {train_output_file}")
    print(f"Validation dataset saved to {val_output_file}")
    print(f"Training images copied to {train_dir}")
    print(f"Validation images copied to {val_dir}")