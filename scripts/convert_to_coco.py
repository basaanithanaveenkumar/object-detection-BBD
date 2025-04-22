import os
import json
from tqdm import tqdm


dir_list = ["data/100k/valid", "data/100k/train", "data/100k/test"]

for dir_path in dir_list:
    # Define paths
    images_dir = dir_path  # Update with your images directory
    annotations_dir = dir_path  # Update with your annotations directory
    output_json_path = dir_path + "/_annotations.coco.json"  # Output COCO format JSON

    # Initialize COCO structure
    coco = {
        "info": {
            "description": "BDD100K Dataset in COCO Format",
            "version": "1.0",
            "year": 2023,
            "contributor": "User",
            "url": "",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id":0,"name":"object_detection_bbd","supercategory":"none"}] # define super catogory
    }
    category_dict = {}
    current_category_id = 1
    annotation_id = 1
    class_list=["car","traffic sign","traffic light","person","truck","bus","bike","rider","motor","train"]
    category_dict = {item: idx + 1 for idx, item in enumerate(class_list)}
    print(category_dict)
    for category_id, category_name in enumerate(class_list,start=1):
        class_dict = {
            "id": category_id,
            "name": category_name,
            "supercategory": "object_detection_bbd"  # Note: Fixed typo from your example
        }
        coco['categories'].append(class_dict)
    # Collect all annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]

    # Process each annotation file with a progress bar
    for annotation_file in tqdm(annotation_files, desc="Processing annotations"):
        annotation_path = os.path.join(annotations_dir, annotation_file)
        with open(annotation_path, "r") as f:
            try:
                bdd_data = json.load(f)
                # print(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {annotation_file}")
                continue

        # Extract image filename from annotation data
        image_filename = bdd_data.get("name", "")
        image_filename = os.path.splitext(annotation_file)[0] + ".jpg"

        # Check if image exists
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image {image_filename} not found, skipping.")
            continue

        # BDD100K images are 1280x720
        width, height = 1280, 720

        # Add image entry to COCO
        image_id = len(coco["images"]) + 1
        coco_image = {
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
        }
        coco["images"].append(coco_image)

        # Process each object in the annotation

        for obj in bdd_data["frames"][0]["objects"]:
            category_name = (
                obj.get("category", "").strip().lower()
            )  # Normalize category name
            if category_name not in class_list:
                continue
            if not category_name:
                continue
            category_id = category_dict[category_name]

            # Extract and validate bounding box 
            bbox = obj.get('box2d', {})
            required_keys = ['x1', 'y1', 'x2', 'y2']
            if not all(key in bbox for key in required_keys):
                continue

            try:
                x1 = float(bbox["x1"])
                y1 = float(bbox["y1"])
                x2 = float(bbox["x2"])
                y2 = float(bbox["y2"])
            except ValueError:
                continue  # Skip invalid bbox values

            # Convert to COCO format [x, y, width, height]
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width <= 0 or bbox_height <= 0:
                continue  # Skip invalid bbox

            coco_bbox = [x1, y1, bbox_width, bbox_height]
            area = bbox_width * bbox_height

            # Create annotation entry
            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
            }
            coco["annotations"].append(coco_annotation)
            annotation_id += 1

    # Save the COCO format JSON
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Conversion complete. COCO format JSON saved to {output_json_path}")
