from unsloth import FastModel
from typing import List, Tuple, Union, Optional, Any, Dict
import math
import torch
import warnings
import os
import re
import torch.nn.functional as F

class EntityClassifier:
    """
    Wrapper for the Chat-LLM. 
    Focuses solely on loading the model and classifying provided prompts.
    """

    def __init__(
        self,
        model_path: str = "Piece-Of-Schmidt/NEClass_location",
        max_seq_length: int = 1024,
        load_in_4bit: bool = True,
        dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
        device: str = "cuda",
        padding_side: str = "left"
    ):
        """
        Loads the LLM. Heavy lifting happens here once.
        """
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        self.device = device
        self.model_path = model_path
        self.padding_side = padding_side

        # Load Model & Tokenizer
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=dtype,
        )
        FastModel.for_inference(self.model)
        self.model.config.use_cache = use_cache
        self.model.to(self.device)
        self.tokenizer.padding_side = padding_side

        # Task definitions mapping
        self._task_instructions = {
            "location": (
                "Your task is to classify the provided Named Entity in terms of its Location, that is, the country/region that this entity is most closely assiciated with. Use the provided surrounding context as decision support whenever available."
            ),
            "ressort": (
                "our task is to classify the provided Named Entity into a ressort label, i.e. the domain that the entity is most closely associated with.\n\nInstructions:\n- Return only one of the following labels: politics, economy, society, sport, location, unk (for unknown).\n- If a country name is used metonymically to represent something else (e.g., a sports team, the government, or the headquarters of a company), assign the corresponding functional label (sport, politics, society, economy). Otherwise, assign the label 'location' (or 'unk').\n- Always use the provided surrounding context as decision support whenever available.\n- In ambiguous cases, choose the most contextually appropriate label. If no decision is possible, return 'unk'."
            )
        }

    def classify_prompts(
        self,
        prompts: List[str],
        batch_size: int = 16,
        max_new_tokens: int = 15,
        include_probabilities: bool = True
    ) -> Tuple[List[str], Optional[List[float]]]:
        """
        Classifies a list of prepared prompts.
        """
        # --- GUARD CLAUSE: Catch empty inputs to prevent TypeError ---
        if not prompts:
            return [], ([] if include_probabilities else None)

        all_labels = []
        all_probs = [] if include_probabilities else None

        # Determine task instruction based on model path (simple heuristic)
        task_instruction = self._task_instructions.get("ressort" if "ressort" in self.model_path else "location")

        # Create Batches
        n_batches = math.ceil(len(prompts) / batch_size)
        
        for b in range(n_batches):
            batch_prompts = prompts[b * batch_size : (b + 1) * batch_size]
            
            # 1. Templating & Tokenization
            texts = []
            for p in batch_prompts:
                msgs = [
                    {"role": "system", "content": task_instruction},
                    {"role": "user", "content": p},
                ]
                # Apply chat template
                full_prompt = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ).removeprefix("<bos>")
                texts.append(full_prompt)

            inputs = self.tokenizer(
                text=texts, 
                return_tensors="pt", 
                padding=True, 
                padding_side=self.padding_side
            ).to(self.device)

            # 2. Inference
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=include_probabilities,
                    return_dict_in_generate=True,
                )

            # 3. Decoding
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[:, input_len:]
            batch_labels = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_labels.extend(batch_labels)

            # 4. Probabilities (optional)
            if include_probabilities:
                # Get log_probs of the first generated token
                first_token_scores = outputs.scores[0] # [batch, vocab]
                log_probs = F.log_softmax(first_token_scores, dim=-1)
                # Get the score of the token that was actually selected (greedy)
                first_token_ids = generated_ids[:, 0]
                selected_log_probs = log_probs.gather(1, first_token_ids.unsqueeze(1)).squeeze(1)
                all_probs.extend(selected_log_probs.exp().tolist())

        return all_labels, all_probs

    def texts_to_paragraphs(self, texts: List[str], max_seq_len: int = 512) -> List[Dict[str, Any]]:
        """
        Splits texts at sentence boundaries. 
        Guarantees that no output chunk exceeds max_seq_len tokens.
        """
        out = []
        for idx, text in enumerate(texts):
            # 1. Tokenize full text
            full_res = self.tokenizer(text=text, add_special_tokens=False)["input_ids"]
            # Ensure flat list
            full_ids = full_res[0] if (len(full_res) > 0 and isinstance(full_res[0], list)) else full_res
            
            if len(full_ids) <= max_seq_len:
                out.append({"idx": idx, "text": text})
                continue

            # 2. Split at linebreak
            segments = re.split(r'(?<=[.!?\n])\s+', text)
            
            current_chunk_ids = []
            
            for seg in segments:
                if not seg.strip(): continue
                
                # Tokenize segment
                res = self.tokenizer(text=seg, add_special_tokens=False)["input_ids"]
                seg_ids = res[0] if (len(res) > 0 and isinstance(res[0], list)) else res
                seg_ids = [int(i) for i in seg_ids] 
                
                seg_len = len(seg_ids)
                
                # If segment is too long on its own
                if seg_len > max_seq_len:
                    
                    if current_chunk_ids:
                        out.append({"idx": idx, "text": self.tokenizer.decode(current_chunk_ids, skip_special_tokens=True).strip()})
                        current_chunk_ids = []
                    
                    # Hard split overly long snippet
                    for i in range(0, seg_len, max_seq_len):
                        sub_ids = seg_ids[i : i + max_seq_len]
                        out.append({"idx": idx, "text": self.tokenizer.decode(sub_ids, skip_special_tokens=True).strip()})
                    continue

                # Check if segment fits
                if len(current_chunk_ids) + seg_len > max_seq_len:
                    out.append({"idx": idx, "text": self.tokenizer.decode(current_chunk_ids, skip_special_tokens=True).strip()})
                    current_chunk_ids = seg_ids
                else:
                    current_chunk_ids.extend(seg_ids)
            
            # return
            if current_chunk_ids:
                out.append({"idx": idx, "text": self.tokenizer.decode(current_chunk_ids, skip_special_tokens=True).strip()})
                
        return out
    

    # def texts_to_paragraphs(self, texts: List[str], max_seq_len: int = 512) -> List[Dict[str, Any]]:
    #     """Helper to split long texts based on token count."""
    #     # Simple splitting logic
    #     out = []
    #     for idx, text in enumerate(texts):
    #         # Very rough estimation. Check length string-wise first to avoid tokenizing short texts
    #         if len(text) < max_seq_len * 2: 
    #             out.append({"idx": idx, "text": text})
    #             continue
                
    #         # Tokenize only if necessary
    #         tokens = self.tokenizer(text=text, add_special_tokens=False)["input_ids"]
    #         if len(tokens) <= max_seq_len:
    #             out.append({"idx": idx, "text": text})
    #         else:
    #             # Chunking
    #             for i in range(0, len(tokens), max_seq_len):
    #                 chunk_ids = tokens[i : i + max_seq_len]
    #                 chunk_text = self.tokenizer.decode(chunk_ids)
    #                 out.append({"idx": idx, "text": chunk_text})
    #     return out

