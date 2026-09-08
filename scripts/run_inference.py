import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from PIL import Image
from inference import load_model, run_single_example, run_inference_on_manifest, save_results
from dataset_construction import load_manifest

model, processor = load_model()
"""image = Image.open("data/val2017/000000000139.jpg").convert("RGB")
result = run_single_example(model, processor, image, "Is there a dog in this image?")

print(result.generated_text)
print(result.parsed_answer)
print(result.confidence)
print(result.hidden_states[1].shape)"""

data_dir = os.path.join(os.path.dirname(__file__), "../data")
manifest = load_manifest(os.path.join(data_dir, "manifest.csv"))

"""debug_manifest = manifest[:45]
results = run_inference_on_manifest(model, processor, debug_manifest, os.path.join(data_dir, "val2017"))

for r in results:
    print(r.image_id, r.question_type, r.ground_truth, r.parsed_answer, r.generated_text)

print(results[0].hidden_states[1].shape)"""

results = run_inference_on_manifest(
    model, processor, manifest, os.path.join(data_dir, "val2017"),checkpoint_path=os.path.join(data_dir, "inference_results.pkl"),checkpoint_every=15,)
save_results(results, os.path.join(data_dir, "inference_results.pkl"))