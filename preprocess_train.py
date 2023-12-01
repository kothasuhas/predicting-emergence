import torch
import pickle
import numpy as np
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
import matplotlib.pyplot as plt

MODEL_NAMES = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-12b",
]

DATASET = "wikipedia"
NUM_SENTENCES = 1000
NUM_TOKENS = 50277

if DATASET == "wikipedia":
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]").shuffle()
    sentences = dataset["text"][:NUM_SENTENCES]
    # truncate each sentence to the first 10-20 words
    sentences = [sentence.split(" ")[:random.randint(10, 20)] for sentence in sentences]
    sentences = [" ".join(sentence) for sentence in sentences]
    print(sentences[:10])
elif DATASET == "glue-rte":
    dataset = load_dataset("glue", "rte", split="train").shuffle()
    sentences = []
    for i in range(NUM_SENTENCES):
        sentences.append(dataset["sentence1"][i] + "\nQuestion: " + dataset["sentence2"][i] + " True or False?\nAnswer:")
    print(sentences[:10])
else:
    raise ValueError(f"Unknown dataset {DATASET}")

probs_per_sentence = np.zeros((NUM_SENTENCES, len(MODEL_NAMES), NUM_TOKENS))

with torch.no_grad():
    for model_num, model_name in enumerate(MODEL_NAMES):
        print('--------Trying model', model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()

        for sentence in sentences:
            print('----------------Trying sentence', sentence)
            # extract distribution over next token from model
            input_ids = tokenizer.encode(sentence, return_tensors="pt").cuda()
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :NUM_TOKENS]
            # extract probability distribution over next token using softmax
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)[0]
            assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).cuda())
            
            probs_per_sentence[sentences.index(sentence)][model_num] = next_token_probs.cpu().numpy()

        del model, tokenizer, outputs, logits, next_token_logits, next_token_probs
        torch.cuda.empty_cache()

# cast to float16 to save memory
probs_per_sentence = probs_per_sentence.astype(np.float16)

# pickle the sentences and probs array
with open(f'{DATASET}.pkl', 'wb') as f:
    pickle.dump((MODEL_NAMES, sentences, probs_per_sentence), f)
