from pycocotools.coco import COCO
import numpy as np
import csv

class COCOSubset:
    def __init__(self, coco, image_dir):
        self.coco = coco
        self.image_dir = image_dir
        self.cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        self.img_ids = coco.getImgIds()

def load_coco_subset(annotation_path: str, image_dir: str) -> "COCOSubset":
    coco = COCO(annotation_path)
    return COCOSubset(coco, image_dir)

def compute_cooccurrence(coco: "COCOSubset") -> dict[tuple[str, str], int]:

    counts = {}
    for img_id in coco.img_ids:
        annotations = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds=img_id))
        categories = set()
        for annotation in annotations:
            categories.add(coco.cat_id_to_name[annotation["category_id"]])

        categories_list = list(categories)


        for i in range(len(categories_list)):
            for j in range(i+1, len(categories_list)):
                pair = (categories_list[i], categories_list[j])
                counts[pair] = counts.get(pair,0)+1

    return counts
            
