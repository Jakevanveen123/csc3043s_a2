import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceResult:
    def __init__(self, image_id, category, question_type, ground_truth, generated_text, parsed_answer, confidence, hidden_states):
        self.image_id = image_id
        self.category = category
        self.question_type = question_type
        self.ground_truth = ground_truth
        self.generated_text= generated_text
        self.parsed_answer = parsed_answer
        self.confidence = confidence
        self.hidden_states = hidden_states

def load_model(model_name: str, device: str):
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)


def run_single_example(model, processor, image, question: str) -> InferenceResult:
    messages = [{"role": "user", "content": [{"type": "image"},{"type": "text", "text": "Can you describe this image?"}]},]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device) 
    return 

def run_inference_on_manifest(model, processor, manifest: list[dict],
image_dir: str) -> list[InferenceResult]:
    return list(1)

def save_results(results: list[InferenceResult], path: str) -> None:
    return None

def load_results(path: str) -> list[InferenceResult]:

    return list(1)