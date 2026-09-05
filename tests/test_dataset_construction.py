import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dataset_construction import load_coco_subset, compute_cooccurrence, load_manifest

data_dir = os.path.join(os.path.dirname(__file__), "../data")
coco = load_coco_subset(os.path.join(data_dir, "instances_val2017.json"), os.path.join(data_dir, "val2017"))
questions = load_manifest(os.path.join(data_dir, "manifest.csv"))

def present_categories(img_id):
    annotations = coco.coco.loadAnns(coco.coco.getAnnIds(imgIds=int(img_id)))
    return set(coco.cat_id_to_name[a["category_id"]] for a in annotations)

counts = {}
seen = set()
duplicates = 0
bad_present = 0
bad_adversarial = 0

for q in questions:
    counts[q["question_type"]] = counts.get(q["question_type"], 0) + 1
    pair = (q["image_id"], q["category"])
    if pair in seen:
        duplicates += 1
    seen.add(pair)
    if q["question_type"] == "present" and q["category"] not in present_categories(q["image_id"]):
        bad_present += 1
    if q["question_type"] == "absent_adversarial" and q["category"] in present_categories(q["image_id"]):
        bad_adversarial += 1

print("counts per type:", counts)
print("duplicate pairs:", duplicates)
print("bad present labels:", bad_present)
print("bad adversarial labels:", bad_adversarial)
print("cooccurrence deterministic:", compute_cooccurrence(coco) == compute_cooccurrence(coco))