import torch

class ResponseGenerator:
    def __init__(self, model, tokenizer, prompt_type="reasoning", temperature=0.8):
        self.model = model
        self.tokenizer = tokenizer
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

    def generate_responses(self, batch, n_responses=30, max_length=150):
        """Generate multiple responses for each query in the batch."""
        responses = []
        for query in batch:
            prompt = self.generate_prompt(query)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(torch.device("cuda"))
            for _ in range(n_responses):
                output = self.model.generate(input_ids, max_length=max_length, temperature=self.temperature, num_return_sequences=1)
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                responses.append((query, response))
        return responses

    def get_embedding(self, queries):
        """Compute embeddings for a batch of queries."""
        input_ids = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True).input_ids.to(torch.device("cuda"))
        embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings.mean(dim=1).mean(dim=0).cpu().numpy()  # Mean of embeddings for the batch
