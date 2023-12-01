import torch
import pickle
import numpy as np
import random
from transformers import (
    AutoTokenizer,
)
import matplotlib.pyplot as plt

##############################################
# Load data
##############################################

TARGET_MODEL_NAME = "EleutherAI/pythia-12b"
NUM_TOKENS = 50277

MODEL_PARAM_DICT = {
    "EleutherAI/pythia-70m": 70,
    "EleutherAI/pythia-160m": 160,
    "EleutherAI/pythia-410m": 410,
    "EleutherAI/pythia-1b": 1000,
    "EleutherAI/pythia-1.4b": 1400,
    "EleutherAI/pythia-2.8b": 2800,
    "EleutherAI/pythia-12b": 12000,
}

DATASET = "wikipedia"
# DATASET = "glue-rte"

# load data
MODEL_NAMES, sentences, probs_per_sentences = pickle.load(open(f"{DATASET}.pkl", "rb"))
NUM_SENTENCES = len(sentences)

print(f"Loaded {len(sentences)} sentences from {DATASET} dataset")
print(f"Loaded {len(MODEL_NAMES)} models")
print(f" - {MODEL_NAMES}")
print(f"Loaded {probs_per_sentences.shape} probabilities per sentence")

assert probs_per_sentences.shape == (NUM_SENTENCES, len(MODEL_NAMES), NUM_TOKENS), f"Invalid shape of {probs_per_sentences.shape}"
assert MODEL_NAMES[-1] == TARGET_MODEL_NAME

##############################################
# Define strategies (add your own here!)
##############################################

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)

# a strategy is a function that takes in
# - a list of sentences
# - a list of model names for probs_per_sentences
# - the name of the target model
# - a numpy array of probabilities per sentence
# and returns a list of guesses per sentence

# dummy strategy: always guess the first token
def dummy_strategy(sentences, model_names, target_model_name, probs_per_sentences):
    guesses_per_sentence = []
    for sentence in sentences:
        guesses_per_sentence.append(0)
    return guesses_per_sentence

# random strategy: always guess a random token
def random_strategy(sentences, model_names, target_model_name, probs_per_sentences):
    guesses_per_sentence = []
    for sentence in sentences:
        guesses_per_sentence.append(random.randint(0, NUM_TOKENS - 1))
    return guesses_per_sentence

# average strategy: always guess the token with highest average probability
def average_strategy(sentences, model_names, target_model_name, probs_per_sentences):
    guesses_per_sentence = []
    for sentence_num, sentence in enumerate(sentences):
        average_probs = np.mean(probs_per_sentences[sentence_num], axis=0)
        guesses_per_sentence.append(np.argmax(average_probs))
    return guesses_per_sentence

# TODO: add your own strategies here!

##############################################
# Evaluate strategies
##############################################

# evaluate top 1 accuracy of a strategy
def evaluate_strategy(strategy, num_samples):
    guess_per_sentence = strategy(sentences[:num_samples], MODEL_NAMES[:-1], TARGET_MODEL_NAME, probs_per_sentences[:num_samples,:-1])
    true_dist_per_sentence = probs_per_sentences[:num_samples, -1]
    true_top1_per_sentence = np.argmax(true_dist_per_sentence, axis=1)
    correct = 0
    for i in range(num_samples):
        if guess_per_sentence[i] == true_top1_per_sentence[i]:
            correct += 1
    return correct / num_samples

# evaluate multiple strategies
def evaluate_strategies(strategies, num_samples):
    for strategy_name, strategy in strategies.items():
        print(f"Evaluating strategy {strategy_name}...")
        print(f"Accuracy: {evaluate_strategy(strategy, num_samples)}")

# define strategies to evaluate
strategies = {
    "dummy": dummy_strategy,
    "random": random_strategy,
    "average": average_strategy,
}

evaluate_strategies(strategies, NUM_SENTENCES)
