---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
license: cc-by-nc-4.0
---

# FinLang/finance-embeddings-investopedia

This is the Investopedia embedding for finance application by the FinLang team. The model is trained using our open-sourced finance dataset from https://huggingface.co/datasets/FinLang/investopedia-embedding-dataset

This is a finetuned embedding model on top of BAAI/bge-base-en-v1.5. It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search in RAG applications.


This project is for research purposes only. Third-party datasets may be subject to additional terms and conditions under their associated licenses.


## Plans
* The research paper will be published soon.
* We are working on a v2 version of the model where we are increasing the training corpus of financial data and using improved techniques for training embeddings.


## Usage (LLamaIndex)

Simply specify the Finlang embedding during the indexing procedure for your Financial RAG applications.

```
from llama_index.embeddings import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="FinLang/investopedia_embedding")
```


## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed (see https://huggingface.co/sentence-transformers):

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('FinLang/investopedia_embedding')
embeddings = model.encode(sentences)
print(embeddings)
```

Example code testing:

```
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("FinLang/investopedia_embedding")

query_1 = "What is a potential concern with allowing someone else to store your cryptocurrency keys, and is it possible to decrypt a private key?"
query_2 = "A potential concern is that the entity holding your keys has control over your cryptocurrency in a custodial relationship. While it is theoretically possible to decrypt a private key, with current technology, it would take centuries or millennia for the 115 quattuorvigintillion possibilities. Most hacks and thefts occur in wallets, where private keys are stored."

embedding_1 = model.encode(query_1)
embedding_2 = model.encode(query_2)
scores = (embedding_1*embedding_2).sum()
print(scores) # 0.862
```



## Evaluation Results

We evaluate our model on unseen pairs of sentences for similarity and unseen shuffled pairs of sentences for dissimilarity. Our evaluation suite contains sentence pairs from: Investopedia (to test for proficiency on finance),
and Gooaq, MSMARCO,stackexchange_duplicate_questions_title_title, yahoo_answers_title_answer (to evaluate models ability to avoid forgetting after finetuning).

## License

Since non-commercial datasets are used for fine-tuning, we release this model as cc-by-nc-4.0.


## Citation [Coming Soon]