from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import random

# Load annotations
dataDir = "data/100k/test/"
dataType = ".coco"
annFile = f"{dataDir}/_annotations{dataType}.json"
coco = COCO(annFile)

# Load image and annotations
img_id = 100  # Example image ID
img = coco.loadImgs(img_id)[0]

# Use local image path
image_path = f"{dataDir}/{img['file_name']}"
I = cv2.imread(image_path)
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

# Get all category information
cats = coco.loadCats(coco.getCatIds())
cat_ids = [cat["id"] for cat in cats]

# Create a color map for each category
colors = {}
for cat_id in cat_ids:
    colors[cat_id] = [random.randint(0, 255) for _ in range(3)]

# Load annotations
annIds = coco.getAnnIds(imgIds=img["id"])
anns = coco.loadAnns(annIds)

# Draw bounding boxes with class-specific colors
for ann in anns:
    x, y, w, h = ann["bbox"]
    cat_id = ann["category_id"]
    color = colors[cat_id]
    cv2.rectangle(I, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

    # Add class label text
    cat = coco.loadCats([cat_id])[0]
    label = cat["name"]
    cv2.putText(
        I, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )

# Display the image
plt.figure(figsize=(12, 8))
plt.imshow(I)
plt.axis("off")
plt.show()
