import numpy as np

class LinUCB:
    def __init__(self, d, K, alpha=0.1):
        self.d = d  # Dimension of feature vectors
        self.K = K  # Number of reward models (arms)
        self.alpha = alpha  # Exploration parameter

        # Initialize the parameters for each arm (reward model)
        self.A = [np.eye(d) for _ in range(K)]  # Covariance matrices for each arm
        self.b = [np.zeros(d) for _ in range(K)]  # Bias vector for each arm
        self.reward_history = []  # To track rewards over time

    def select_arm(self, context):
        # Compute the upper confidence bound for each arm
        upper_bounds = []
        for k in range(self.K):
            A_inv = np.linalg.inv(self.A[k])
            theta_k = A_inv @ self.b[k]
            mean = context.T @ theta_k
            uncertainty = self.alpha * np.sqrt(context.T @ A_inv @ context)
            upper_bounds.append(mean + uncertainty)

        # Select the arm with the highest upper confidence bound
        return np.argmax(upper_bounds)

    def normalize_reward(self, reward):
        """Normalize the reward based on the history using quantiles."""
        if len(self.reward_history) < 2:
            return reward

        # Calculate 20th and 80th quantiles
        q_lo = np.percentile(self.reward_history, 20)
        q_hi = np.percentile(self.reward_history, 80)

        if reward < q_lo:
            return 0
        elif reward > q_hi:
            return 1
        else:
            return (reward - q_lo) / (q_hi - q_lo)

    def update(self, chosen_arm, context, reward):
        # Track reward history
        self.reward_history.append(reward)

        # Normalize the reward based on quantiles
        normalized_reward = self.normalize_reward(reward)

        # Update parameters based on normalized reward and context
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += normalized_reward * context