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
        categories_list = sorted(categories)
        for i in range(len(categories_list)):
            for j in range(i+1, len(categories_list)):
                pair = (categories_list[i], categories_list[j])
                counts[pair] = counts.get(pair,0)+1

    return counts

def sample_image_ids(coco, n_images, seed):
    rng = np.random.default_rng(seed=seed)
    return rng.choice(coco.img_ids, size=n_images, replace=False).tolist()

def build_question_set(coco, image_ids, cooccurrence, seed):
    rng = np.random.default_rng(seed=seed)
    all_categories = list(coco.cat_id_to_name.values())
    questions = []

    for img_id in image_ids:
        annotations = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds=img_id))
        categories = set()
        for annotation in annotations:
            categories.add(coco.cat_id_to_name[annotation["category_id"]])
        categories_list = sorted(categories)

        present_cat = categories_list[rng.integers(len(categories_list))]
        questions.append({"image_id": img_id, "category": present_cat,
                           "question": f"Is there a {present_cat} in this image?",
                           "question_type": "present", "ground_truth": "yes"})

        best_value = -1
        adversary_category = None
        best_present_partner = None
        for key, value in cooccurrence.items():
            if key[0] in categories_list and key[1] not in categories_list:
                absent_side = key[1]
                present_side = key[0]
            elif key[1] in categories_list and key[0] not in categories_list:
                absent_side = key[0]
                present_side = key[1]
            else:
                continue
            if value > best_value:
                best_value = value
                adversary_category = absent_side
                best_present_partner = present_side

        print(img_id, best_present_partner, adversary_category, best_value)
        

        questions.append({"image_id": img_id, "category": adversary_category,
                           "question": f"Is there a {adversary_category} in this image?",
                           "question_type": "absent_adversarial", "ground_truth": "no"})

        categories_not_in_image = []
        for c in all_categories:
            if c not in categories_list and c not in adversary_category:
                categories_not_in_image.append(c)
        
        random_cat = categories_not_in_image[rng.integers(len(categories_not_in_image))]
        questions.append({"image_id": img_id, "category": random_cat,
                           "question": f"Is there a {random_cat} in this image?",
                           "question_type": "absent_random", "ground_truth": "no"})

    return questions

def save_manifest(questions, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "category", "question", "question_type", "ground_truth"])
        writer.writeheader()
        writer.writerows(questions)

def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))



                    


            

        
            
