import torch
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
import pickle
import time

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
        _attn_implementation="sdpa" if DEVICE == "cuda" else "eager",
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
    answer_start = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=5,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = gen_out.sequences
    answer_ids = generated_ids[0, answer_start:]
    generated_text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)

    lowered = generated_text.strip().lower()
    if "yes" in lowered:
        parsed_answer = True
    elif "no" in lowered:
        parsed_answer = False
    else:
        parsed_answer = None


    step_scores = torch.stack(gen_out.scores, dim=1)[0]  
    probs = torch.softmax(step_scores, dim=-1)
    token_probs = probs[range(len(answer_ids)), answer_ids]
    confidence = token_probs.mean().item()

    num_layers = len(gen_out.hidden_states[0])
    hidden_states = {}
    for layer_idx in range(num_layers):
        parts = [gen_out.hidden_states[0][layer_idx][0]]  
        parts += [step[layer_idx][0] for step in gen_out.hidden_states[1:]]  
        hidden_states[layer_idx] = torch.cat(parts, dim=0).cpu().numpy().astype("float16")

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


def run_inference_on_manifest(model, processor, manifest, image_dir, checkpoint_path=None, checkpoint_every=25):
    completed = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            while True:
                try:
                    result = pickle.load(f)
                except EOFError:
                    break
                completed.add((result.image_id, result.question_type))

    results = []
    start = time.time()

    for i, row in enumerate(manifest):
        image_id = int(row["image_id"])
        if (image_id, row["question_type"]) in completed:
            continue
        filename = f"{image_id:012d}.jpg"
        image = Image.open(os.path.join(image_dir, filename)).convert("RGB")

        result = run_single_example(model, processor, image, row["question"])
        result.image_id = image_id
        result.category = row["category"]
        result.question_type = row["question_type"]
        result.ground_truth = row["ground_truth"] == "yes"

        results.append(result)

        if (i + 1) % checkpoint_every == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(manifest) - (i + 1)) / rate
            print(f"{i + 1}/{len(manifest)} done, {elapsed / 60:.1f} min elapsed, {remaining / 60:.1f} min remaining")
            if checkpoint_path:
                save_results(results, checkpoint_path)
                results = []

    if checkpoint_path and results:
        save_results(results, checkpoint_path)

    return results

def save_results(results, path):
    with open(path, "ab") as f:
        for result in results:
            pickle.dump(result, f)

def load_results(path):
    results = []
    with open(path, "rb") as f:
        while True:
            try:
                results.append(pickle.load(f))
            except EOFError:
                break
    return results
