import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "clip")))
import torch
import numpy as np
import pandas as pd
from clip import CLIPModelWrapper
import time

start_time = time.time()

SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)

clip_model = CLIPModelWrapper(model_name="./clip")

def get_prompt(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](str(word)) for word in words]
    return prompt

nouns = pd.read_csv("./wordnet/WordNetNouns.csv", encoding='gbk')
if "word" not in nouns.columns or "definition" not in nouns.columns:
    raise ValueError("Dont find 'word' or 'definition' columns")
print("nouns shape", len(nouns))

nouns = nouns.dropna(subset=['word']).drop_duplicates(subset=['word'], keep='first')
print("nouns shape", len(nouns))

nouns_num = nouns.shape[0]
batch_size = 2048

for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
    template_start = time.time()
    features = []
    print("Inferring text features for index", index)
    for i in range(nouns_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > nouns_num:
            end = nouns_num
        nouns_batch = nouns[start:end]
        print(f"Processing batch {i}: Start={start}, End={end}, Batch size={len(nouns_batch)}")
        words = [str(word) for word in nouns_batch['word'] if pd.notnull(word)]
        with torch.no_grad():
            prompt = get_prompt(words, index)
            print(f"Generated prompts: {prompt[:5]}")
            feature = clip_model.encode_text(prompt)
            features.append(feature.cpu().numpy())
        print(f"[Completed {(i+1) * batch_size}/{nouns_num}]")
    features = np.concatenate(features, axis=0)
    print("Feature shape:", features.shape)
    np.save(f"./wordnet/wordnet_embedding_prompt_{index}.npy", features)
    template_end = time.time()
    print(f"Template {index} finished in {template_end - template_start:.2f} seconds")

embeddings = np.zeros((nouns_num, 512))
for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
    embedding = np.load(f"./wordnet/wordnet_embedding_prompt_{index}.npy")
    embeddings += embedding
embeddings = embeddings / len(SIMPLE_IMAGENET_TEMPLATES)
np.save("./wordnet/wordnet_embedding_ensemble.npy", embeddings)
end_time = time.time()
total_duration = end_time - start_time

print("---" * 10)
print(f"Over!")
print(f"Total Time: {total_duration:.2f} s")
print(f"Total Time: {total_duration / 60:.2f} min")