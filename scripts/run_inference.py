import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from PIL import Image
from inference import load_model, run_single_example

model, processor = load_model()
image = Image.open("data/val2017/000000000139.jpg").convert("RGB")
result = run_single_example(model, processor, image, "Is there a dog in this image?")

print(result.generated_text)
print(result.parsed_answer)
print(result.confidence)
print(result.hidden_states[1].shape)