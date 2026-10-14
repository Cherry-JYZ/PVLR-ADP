import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "/home/jiayingzi/DMNet/model/Qwen3-VL-2B-Instruct"
IMAGE_PATH = "/home/jiayingzi/DMNet/2.jpg"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# prompt = """Focus on spatial layout for image segmentation.
# For each visible object, describe:
# - category
# - approximate position (left/right/top/bottom/center)
# - spatial relation to other objects (adjacent, surrounding, crossing)
# - shape or extent (elongated, compact, large area, small area)

# Output concise factual phrases.
# Avoid scene narration.
# """
prompt = "Describe the specific style and content information of the Chinese mural images."

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE_PATH,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)


with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=32,   
        do_sample=False      
    )


generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print("\n===== Qwen3-VL Spatial Description =====")
for line in output_text:
    print(line)
