# NEClass: A Lightweight LLM Pipeline for Context-Dependent Entity Classification

**NEClass** (Named Entity Classification) is a specialized NLP pipeline designed to bridge the gap between coarse-grained Named Entity Recognition (NER) and fine-grained, context-sensitive classification. While standard NER systems only identify categories like `PER` or `LOC`, NEClass uses fine-tuned Large Language Models (LLMs) to assign entities to specific labels (e.g., country affiliation or thematic ressort) based on their surrounding context.

### 🧠 Specialized Models

The pipeline is designed to work exclusively with the following fine-tuned models from [Hugging Face](https://huggingface.co/Piece-Of-Schmidt). Choose the model that fits your research question:

| Model Task | Path | Use Case |
| --- | --- | --- |
| **Location Classification** | `Piece-Of-Schmidt/NEClass_location` | Use this to map entities (like "Dubai" or "The White House") to their respective **geographical or political country/region**. |
| **Ressort Classification** | `Piece-Of-Schmidt/NEClass_ressort` | Use this to classify entities into **thematic domains** (e.g., politics, economy, sport) based on their functional role in the text. |

*Note: The pipeline logic (prompting & post-processing) is specifically optimized for these Gemma-3-4b-it based models.*

## 🚀 Key Features

* **Context-Aware**: Distinguishes between entities based on their role in a sentence (e.g., "Germany" as a geographical location vs. "Germany" as a political actor in a sports or political context).
* **Lightweight & Fast**: Optimized with **Unsloth** and **4-bit quantization** for high-speed inference on consumer-grade GPUs.
* **End-to-End Pipeline**: Handles intelligent text splitting (preserving sentence boundaries), NER extraction, and LLM classification in a single call.
* **NER-Agnostic**: While the classifier is fixed, the initial extraction is compatible with most Hugging Face NER models (defaulting to RoBERTa-multilingual).

---

## 📦 Installation

```bash
# 1. Install optimized dependencies
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 2. Install NEClass
pip install git+https://github.com/Piece-Of-Schmidt/NEClass.git

```

---

## 🛠 Usage

NEClass is designed for simplicity. Simply change parameters like `context_size` or `batch_size` on the fly. Long documents are automatically split so they fit in the NER model's context window (default: 512 tokens).

```python
from neclass import NECPipeline

# Initialize the pipeline
pipe = NECPipeline(
    model_path="Piece-Of-Schmidt/NEClass_location", # Fine-tuned Classifier
    ner_model="julian-schelb/roberta-ner-multilingual" # Base NER
)

texts = [
    "Angela Merkel enjoys eating fish buns with Emmanuel Macron in Dubai.",
    "The match between Tunisia and Morocco ended 5:1."
]

# Run inference
results = pipe(
    texts, 
    context_size=80, # n characters (left and right of identified NE) that are included in LLM prompt for classification
    batch_size=16,
    include_probabilities=True
)

import pandas as pd
print(pd.DataFrame(results))

```

---

## 📝 Methodology & Paper

NEClass combines a multilingual RoBERTa-based NER model with a LoRA-fine-tuned **Gemma** LLM. By injecting a context window of  characters around an entity into the prompt, the model achieves significantly higher accuracy in ambiguous cases compared to zero-shot baselines.

### Citation

If you use NEClass in your research, please cite:

```bibtex
@article{schmidt2024neclass,
  title={NEClass: A Lightweight LLM Pipeline for Context-Dependent Named Entity Classification},
  author={Schmidt, Tobias and Hornig, Nico},
  journal={Draft Version / TU Dortmund University},
  year={2026}
}

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

