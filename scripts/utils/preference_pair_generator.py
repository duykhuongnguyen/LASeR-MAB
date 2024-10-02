class PreferencePairGenerator:
    def __init__(self, reward_model):
        self.reward_model = reward_model

    def generate_preference_pairs(self, responses, P=10):
        """Generate preference pairs by scoring and ranking the responses."""
        # Generate scores for each response
        scores = [(response, self.reward_model.score(query, response)) for query, response in responses]

        # Sort responses based on scores (highest to lowest)
        sorted_responses = sorted(scores, key=lambda x: x[1], reverse=True)

        # Create P preference pairs
        pairs = []
        for i in range(len(sorted_responses)):
            for j in range(i + 1, len(sorted_responses)):
                if len(pairs) < P:
                    pairs.append((sorted_responses[i][0], sorted_responses[j][0]))
                else:
                    break
            if len(pairs) >= P:
                break
        return pairs