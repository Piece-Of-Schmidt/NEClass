from .neclass import EntityClassifier
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import pipeline
import torch

class NECPipeline:
    """
    End-to-End Pipeline: Text -> NER -> LLM Classification.
    """

    def __init__(
        self,
        model_path: str = "Piece-Of-Schmidt/NEClass_location",
        ner_model: Union[str, Any] = "julian-schelb/roberta-ner-multilingual",
        device: str = "cuda",
        load_in_4bit: bool = True,
        use_cache: bool = True,
        max_seq_length: int = 1024
    ):
        """Initializes the models."""
        self.device = device
        
        # 1. Load NER Pipeline
        ner_device = 0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1
        
        if isinstance(ner_model, str):
            self.ner_pipe = pipeline(
                task="ner",
                model=ner_model,
                tokenizer=ner_model,
                aggregation_strategy="simple",
                device=ner_device,
            )
            self.is_custom_ner = False
        else:
            # Custom function / object provided by user
            self.ner_pipe = ner_model
            self.is_custom_ner = True

        # 2. Load LLM Classifier
        self.clf = EntityClassifier(
            model_path=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            use_cache=use_cache,
            device=device
        )

    def __call__(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        context_size: int = 80,
        min_entity_length: int = 2,
        merge_adjacent: bool = True,
        split_long_texts: bool = True,
        include_probabilities: bool = True,
        return_prompts: bool = False,
        ner_keys: Dict[str, str] = None, 
        **kwargs
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[str]]]:
        
        # 1. Key Mapping Setup
        default_keys = {
            "word": "word",
            "start": "start",
            "end": "end",
            "label": "entity_group",
            "score": "score"
        }
        keys = default_keys.copy()
        if ner_keys:
            keys.update(ner_keys)
        
        if isinstance(texts, str):
            texts = [texts]

        # A. Preprocessing (Text Splitting)
        if split_long_texts:
            segments = self.clf.texts_to_paragraphs(texts)
            processing_texts = [s["text"] for s in segments]
            doc_ids = [s["idx"] for s in segments]
        else:
            processing_texts = texts
            doc_ids = list(range(len(texts)))

        # B. NER Extraction
        ner_results = self.ner_pipe(processing_texts)

        # C. Prepare LLM Prompts
        prompts = []
        flat_entities = []

        prompt_template = "# Context:\n{}\n\n# Entity:\n{}"

        for doc_id, text, entities in zip(doc_ids, processing_texts, ner_results):
            if not entities:
                continue

            if merge_adjacent:
                entities = self._merge_entities(entities, keys)

            for ent in entities:
                raw_word = ent.get(keys["word"], "")
                if not raw_word or len(raw_word) < min_entity_length:
                    continue
                
                word = raw_word.strip()
                try:
                    e_start = int(ent.get(keys["start"], 0))
                    e_end   = int(ent.get(keys["end"], 0))
                except (ValueError, TypeError):
                    continue 
                
                # Context Window Logic
                start = max(0, e_start - context_size)
                end = min(len(text), e_end + context_size)
                context_text = text[start:end]

                # Build Prompt
                prompts.append(prompt_template.format(context_text, word))
                
                # Metadata
                ent["doc_id"] = doc_id 
                flat_entities.append(ent)

        # D. LLM Classification
        if prompts:
            labels, probs = self.clf.classify_prompts(
                prompts, 
                batch_size=batch_size, 
                include_probabilities=include_probabilities
            )
            
            for ent, label in zip(flat_entities, labels):
                ent["label"] = label
            
            if include_probabilities and probs:
                for ent, p in zip(flat_entities, probs):
                    ent["label_prob"] = p

        return (flat_entities, prompts) if return_prompts else flat_entities

    @staticmethod
    def _merge_entities(
        entities: List[Dict[str, Any]], 
        keys: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Merges adjacent entities safely. 
        Uses provided keys or defaults to standard HF format if keys is None.
        """
        if not entities:
            return []
        
        # Fallback: Use Defaults if no Keys provided
        if keys is None:
            keys = {
                "word": "word", 
                "start": "start", 
                "end": "end"
            }
        
        # Mapping Helpers
        k_start = keys.get("start", "start")
        k_end   = keys.get("end", "end")
        k_word  = keys.get("word", "word")

        merged = []
        curr = entities[0].copy()
        
        for next_ent in entities[1:]:
            # Check if end of 'curr' == start of 'next_ent'
            try:
                curr_end = int(curr.get(k_end, 0))
                next_start = int(next_ent.get(k_start, 0))
            except (ValueError, TypeError):
                # Fallback if corrupt
                merged.append(curr)
                curr = next_ent.copy()
                continue

            if next_start == curr_end: 
                # Merge Content
                w1 = curr.get(k_word, "")
                w2 = next_ent.get(k_word, "")
                curr[k_word] = str(w1) + str(w2)
                
                # Update End Index
                curr[k_end] = next_ent.get(k_end)
                
                # Optional: Score Handling (e.g., mean, max -> keep for later versions)
                # if "score" in curr and "score" in next_ent:
                #    curr["score"] = max(curr["score"], next_ent["score"])
            else:
                merged.append(curr)
                curr = next_ent.copy()
        
        merged.append(curr)

        return merged
