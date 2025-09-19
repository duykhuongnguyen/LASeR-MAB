from collections import defaultdict


class PreferencePairGenerator:
    def __init__(self, reward_model):
        self.reward_model = reward_model

    def generate_preference_pairs(self, responses, P=10):
        """Generate preference pairs by scoring and ranking the responses."""
        grouped = defaultdict(list)
        for item in responses:
            grouped[(item["query"], item["prompt"])].append(item["response"])

        pairs = []
        for (query, prompt), candidate_responses in grouped.items():
            if len(candidate_responses) < 2:
                continue

            scored = [
                {
                    "response": response,
                    "score": self.reward_model.score(prompt, response),
                }
                for response in candidate_responses
            ]
            scored.sort(key=lambda item: item["score"], reverse=True)

            pair_count = 0
            for i in range(len(scored)):
                for j in range(i + 1, len(scored)):
                    pairs.append(
                        {
                            "query": query,
                            "prompt": prompt,
                            "chosen": scored[i]["response"],
                            "rejected": scored[j]["response"],
                            "chosen_score": scored[i]["score"],
                            "rejected_score": scored[j]["score"],
                        }
                    )
                    pair_count += 1
                    if P is not None and pair_count >= P:
                        break
                if P is not None and pair_count >= P:
                    break

        return pairs

