"""
Konvertiert YOLO-Format zu COCO-Format für DEIM-Training.
YOLO: class_id x_center y_center width height (normalisiert)
COCO: {image_id, category_id, bbox: [x, y, w, h], area, etc.}
"""

import os
import json
from pathlib import Path
from PIL import Image


def yolo_to_coco(yolo_label_path, image_width, image_height):
    """Konvertiere eine einzelne YOLO-Label-Datei zu COCO-Format."""
    annotations = []
    if not os.path.exists(yolo_label_path):
        return annotations

    with open(yolo_label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = [float(x) for x in parts[1:]]

            # YOLO (normalized) → COCO (absolute)
            x = (x_center - width / 2) * image_width
            y = (y_center - height / 2) * image_height
            w = width * image_width
            h = height * image_height

            annotations.append({
                'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                'area': round(w * h, 2),
                'bbox_format': 'xywh',
                'iscrowd': 0,
                'image_id': None,  # wird später gesetzt
                'category_id': class_id,
            })

    return annotations


def convert_dataset(data_yaml_path='dataset/src/large/data.yaml', output_dir='dataset/src/large'):
    """Konvertiert das gesamte Dataset von YOLO zu COCO."""

    # 1. data.yaml lesen
    with open(data_yaml_path, 'r') as f:
        import yaml
        data = yaml.safe_load(f)

    nc = data['nc']
    names = data['names']
    images_dir = Path(output_dir) / data.get('train', 'images')

    # 2. Alle Bilder durchgehen
    image_files = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.jpeg')) + sorted(images_dir.glob('*.png'))

    images = []
    annotations = []
    category_id_map = {}  # YOLO class_id → COCO category_id

    # Kategorien erstellen (basierend auf den Klassen im data.yaml)
    categories = []
    for cls_id, cls_name in enumerate(names):
        categories.append({
            'id': cls_id,
            'name': cls_name,
            'supercategory': 'object',
        })
        category_id_map[cls_id] = cls_id

    annotation_id = 1
    for img_idx, img_path in enumerate(image_files):
        img = Image.open(img_path)
        w, h = img.size
        img_id = img_idx

        images.append({
            'id': img_id,
            'file_name': img_path.name,
            'width': w,
            'height': h,
        })

        # Korrespondierende Label-Datei finden
        label_path = Path(output_dir) / 'labels' / (img_path.stem + '.txt')
        anns = yolo_to_coco(str(label_path), w, h)

        for ann in anns:
            ann['id'] = annotation_id
            ann['image_id'] = img_id
            annotation_id += 1
            annotations.append(ann)

    # 3. COCO JSON erstellen
    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }

    # 4. In train/val aufteilen
    split_dir = Path(output_dir)
    train_dir = split_dir / 'coco_train'
    val_dir = split_dir / 'coco_val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    # Bilder nach train/val kopieren
    split_idx = int(len(image_files) * 0.8)  # 80/20 Split

    for img in images[:split_idx]:
        src = images_dir / img['file_name']
        if src.exists():
            import shutil
            shutil.copy(src, train_dir / img['file_name'])

    for img in images[split_idx:]:
        src = images_dir / img['file_name']
        if src.exists():
            import shutil
            shutil.copy(src, val_dir / img['file_name'])

    # COCO JSON für train und val erstellen
    train_annotations = [a for a in annotations if a['image_id'] < split_idx]
    val_annotations = [a for a in annotations if a['image_id'] >= split_idx]

    train_coco = {
        'images': [img for img in images[:split_idx]],
        'annotations': train_annotations,
        'categories': categories,
    }

    val_coco = {
        'images': [img for img in images[split_idx:]],
        'annotations': val_annotations,
        'categories': categories,
    }

    train_json_path = train_dir / 'instances_train.json'
    val_json_path = val_dir / 'instances_val.json'

    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f, indent=2)
    with open(val_json_path, 'w') as f:
        json.dump(val_coco, f, indent=2)

    print(f"✅ Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"✅ Val:   {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations")
    print(f"✅ Klassen: {nc}")
    print(f"✅ Train JSON: {train_json_path}")
    print(f"✅ Val JSON:   {val_json_path}")

    return train_json_path, val_json_path


if __name__ == '__main__':
    convert_dataset()