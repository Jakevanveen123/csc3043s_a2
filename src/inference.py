import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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

def load_model():
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.float32,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    return model, processor


def run_single_example(model, processor, image, question: str) -> InferenceResult:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=10)
    answer_start = inputs["input_ids"].shape[1]
    answer_ids = generated_ids[0, answer_start:]
    generated_text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)

    lowered = generated_text.strip().lower()
    if "yes" in lowered:
        parsed_answer = True
    elif "no" in lowered:
        parsed_answer = False
    else:
        parsed_answer = None

    with torch.no_grad():
        outputs = model(input_ids=generated_ids, pixel_values=inputs["pixel_values"], output_hidden_states=True)

    logits = outputs.logits[0, answer_start - 1:-1]
    probs = torch.softmax(logits, dim=-1)
    token_probs = probs[range(len(answer_ids)), answer_ids]
    confidence = token_probs.mean().item()

    hidden_states = {}
    for layer_idx in range(len(outputs.hidden_states)):
        layer_tensor = outputs.hidden_states[layer_idx]
        hidden_states[layer_idx] = layer_tensor[0].cpu().numpy()

    return InferenceResult(
        image_id=None,
        category=None,
        question_type=None,
        ground_truth=None,
        generated_text=generated_text,
        parsed_answer=parsed_answer,
        confidence=confidence,
        hidden_states=hidden_states,
    )


def run_inference_on_manifest(model, processor, manifest: list[dict],image_dir: str) -> list[InferenceResult]:
    return list(1)

def save_results(results: list[InferenceResult], path: str) -> None:
    return None

def load_results(path: str) -> list[InferenceResult]:

    return list(1)