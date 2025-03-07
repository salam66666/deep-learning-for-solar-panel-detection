import os
import json
import cv2
import numpy as np

# 路径设置
images_dir = 'dataset/images'
annotations_dir = 'dataset/annotations'

# 加载数据
def load_data(images_dir, annotations_dir):
    images = []
    labels = []

    for annotation_file in os.listdir(annotations_dir):
        # 加载 JSON 标注
        with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
            annotation = json.load(f)

        # 提取图片路径和标注
        image_name = annotation["image_name"]  # 假设 JSON 包含图片名称
        hotspots = annotation["hotspots"]     # 假设 JSON 包含热点坐标

        # 加载图片
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            images.append(image)
            labels.append(hotspots)  # 记录热点区域
    return images, labels

images, labels = load_data(images_dir, annotations_dir)
print(f"Loaded {len(images)} images and {len(labels)} labels.")
