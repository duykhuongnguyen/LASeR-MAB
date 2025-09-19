import torch

class ResponseGenerator:
    def __init__(self, model, tokenizer, device, prompt_type="reasoning", temperature=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_type = prompt_type
        self.temperature = temperature

    def generate_prompt(self, query):
        """Generate prompt based on the task type."""
        if self.prompt_type == "reasoning":
            return (f"Your task is to answer the question below. "
                    f"Give step-by-step reasoning before you answer, "
                    f"and when you’re ready to answer, please use the format `Final answer:...`.\n"
                    f"Question: {query}\nSolution: ")
        else:
            raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

    def _base_model(self):
        """Return the underlying model, unwrapping DataParallel if needed."""
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def generate_responses(self, batch, n_responses=30, max_new_tokens=150):
        """Generate multiple responses for each query in the batch."""
        responses = []
        model = self._base_model()
        model.eval()
        with torch.no_grad():
            for query in batch:
                prompt = self.generate_prompt(query)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": self.temperature,
                    "num_return_sequences": n_responses,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                outputs = model.generate(**inputs, **generation_kwargs)
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True)
                    if text.startswith(prompt):
                        text = text[len(prompt):].strip()
                    responses.append({
                        "query": query,
                        "prompt": prompt,
                        "response": text.strip(),
                    })
        model.train()
        self.model.train()
        return responses

    def get_embeddings(self, queries, strategy="mean_input"):
        """Compute per-query embeddings with configurable aggregation strategy.

        Parameters
        ----------
        queries : list[str]
            Natural-language prompts to embed.
        strategy : str, optional
            Controls how embeddings are derived.
            - ``"mean_input"`` (default): mean over input-token embeddings.
            - ``"last_input"``: last-token input embedding.
            - ``"mean_hidden"``: mean over final-layer hidden states.
            - ``"last_hidden"``: last-token final-layer hidden state.
        """

        strategy = strategy.lower()
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        base_model = self._base_model()
        was_training = base_model.training
        base_model.eval()

        with torch.no_grad():
            if strategy in {"mean_input", "last_input"}:
                embedding_layer = base_model.get_input_embeddings()
                token_embeddings = embedding_layer(inputs["input_ids"])
                if strategy == "mean_input":
                    features = token_embeddings.mean(dim=1)
                else:  # "last_input"
                    features = token_embeddings[:, -1, :]
            elif strategy in {"mean_hidden", "last_hidden"}:
                outputs = base_model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]
                if strategy == "mean_hidden":
                    features = hidden_states.mean(dim=1)
                else:  # "last_hidden"
                    features = hidden_states[:, -1, :]
            else:
                raise ValueError(f"Unsupported embedding strategy: {strategy}")

        if was_training:
            base_model.train()
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.train()

        return features.detach().cpu().numpy()
