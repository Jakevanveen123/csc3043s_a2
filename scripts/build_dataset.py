import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dataset_construction import load_coco_subset, compute_cooccurrence, sample_image_ids, build_question_set, save_manifest

student_number = "VVNJAK001"
seed = int.from_bytes(student_number.encode("utf-8"), byteorder="big")
print(seed)

data_dir = os.path.join(os.path.dirname(__file__), "../data")
coco = load_coco_subset(os.path.join(data_dir, "instances_val2017.json"), os.path.join(data_dir, "val2017"))
cooccurrence = compute_cooccurrence(coco)
image_ids = sample_image_ids(coco, 200, seed)
questions = build_question_set(coco, image_ids, cooccurrence, seed)
save_manifest(questions, os.path.join(data_dir, "manifest.csv"))