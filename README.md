# NEClass: A Lightweight LLM Pipeline for Context-Dependent Entity Classification

**NEClass** is a specialized NLP pipeline designed to bridge the gap between coarse-grained Named Entity Recognition (NER) and fine-grained, context-sensitive classification. While standard NER systems only identify categories like `PER` or `LOC`, NEClass uses fine-tuned Large Language Models (LLMs) to assign entities to specific labels (e.g., country affiliation or thematic ressort) based on their surrounding context.

## 🚀 Key Features

* **Context-Aware**: Distinguishes between entities based on their role in a sentence (e.g., "Germany" as a location vs. "Germany" as a political actor).
* **Lightweight & Fast**: Optimized with **Unsloth** and **4-bit quantization** for high-speed inference on consumer-grade GPUs.
* **End-to-End Pipeline**: Handles text splitting, NER extraction, context windowing, and LLM classification in a single call.
* **Flexible**: Compatible with any Hugging Face NER model and various fine-tuned LLMs.

---

## 📦 Installation

NEClass is optimized for Linux and Google Colab environments with NVIDIA GPUs.

```bash
# 1. Install optimized dependencies
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 2. Install NEClass
pip install git+https://github.com/YOUR_USERNAME/neclass.git

```

---

## 🛠 Usage

NEClass is designed for simplicity. You can change parameters like `context_size` or `batch_size` on the fly without reloading the models.

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
    context_size=80, 
    batch_size=16, 
    include_probabilities=True
)

import pandas as pd
print(pd.DataFrame(results))

```

### Advanced: Handling Large Documents

NEClass includes a smart splitter that ensures texts fit into the LLM's context window without cutting words or entities in half:

```python
results = pipe(long_document_list, split_long_texts=True, max_seq_len=512)

```

---

## 📊 Models

Models are available on [Hugging Face](https://huggingface.co/Piece-Of-Schmidt):

| Model Task | Base Model | Path |
| --- | --- | --- |
| **Location Classification** | Gemma-3-4b-it | `Piece-Of-Schmidt/NEClass_location` |
| **Ressort Classification** | Gemma-3-4b-it | `Piece-Of-Schmidt/NEClass_ressort` |

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

