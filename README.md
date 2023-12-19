# Predicting Emergence

_For a visual introduction, please look at [these introduction slides](./introduction.pdf)_

People consistently debate whether language models display "emergent abilities", but it's difficult to define the phenomenon in the first place. One simple definition is whether a model's next token distribution predictably changes as we scale up the model's parameters and keep everything else constant. Interestingly, this lends itself to a clean emergence test: can one predict a larger model's activations from smaller models with the same training algorithm and data distribution? 

In this challenge, you will put this definition to the test! Thanks to [EleutherAI](https://www.eleuther.ai/)'s heroic effort with the [Pythia](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) model series, we have access to language models trained across various parameter counts using the same training and data recipe. We formulate our concrete problem as the following

**"Given the next-token probability distribution for a prefix for Pythia-{70m, 160m, 410m, 1b, 1.4b, 2.8b}, can you predict the most likely token given by Pythia-12b?"**

<p align="center"><img width="623" alt="Screen Shot 2023-11-30 at 6 10 12 PM" src="https://github.com/kothasuhas/predicting_emergence/assets/38450656/4cd4c7b7-48aa-48f1-a979-3b5d13c10041"></p>

This repo provides such datasets for 100 prefixes of the [GLUE RTE dataset](https://huggingface.co/datasets/glue/viewer/rte) and 100 prefixes of the [Wikipedia dataset](https://huggingface.co/datasets/wikipedia). These datasets are provided as pickle files `glue-rte.pkl` and `wikipedia.pkl`. Each pickle file contains a tuple of the following information
- `model_names`: names of the models we have logits for (our files have 7)
- `sentences`: all the prefixes we consider (our files have 100)
- `probs_per_sentence`: a `(NUM_SENTENCES x NUM_MODELS x NUM_TOKENS)` array, where `probs_per_sentence[i][j][k]` has, for prefix `sentences[i]`, the probability `model_names[j]` assigns token `k` (our arrays are 100 x 7 x 50277)

For the specific construction process and format, refer to how they are generated/saved in `preprocess_train.py` and how they are loaded in `challenge.py`. 

If you think you have a function that can predict the next-token prediction distribution, put your idea to the test by writing a `strategy` in `challenge.py`! If you're happy with the train performance of your method, email the author at `suhask@andrew.cmu.edu` for a held-out test distribution. After running this challenge at my research group's weekly meeting, I have seen some initial strategies and the "SOTA" strategies happen to be quite heuristic. If you perform better than these strategies, I have some available prizes; if you perform particularly strong, we can get a research paper out of this :)) Note that you shouldn't feel limited to the scope, and I'd be excited to see changes that involve changing/increasing the training data or altering the prediction objective in a compelling way.

This is meant to be a hard (and likely impossible) challenge, so don't get upset if you make little progress. Hopefully, this builds some intuition for how language models behave across scale in a fun challenge. Best of luck!

## Setup


Running `challenge.py` requires only a few standard packages such as `torch`, `huggingface`, etc. All code in this repository was executed with the following conda environment
```
conda env create -f environment.yml
```

[Optional] This repo provides two next-token distribution datasets in pickle files. If you want to generate your own, modify `preprocess_train.py` and run

```
python preprocess_train.py
```

To test a strategy, add it to `challenge.py`, specify the desired pickle file in `DATASET`, and run

```
python challenge.py
```
