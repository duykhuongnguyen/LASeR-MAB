import os
from datasets import load_dataset

class DatasetManager:
    def __init__(self, dataset_paths, test_size=0.2, dev_size=0.25, prompt_type="reasoning"):
        """
        Initializes the DatasetManager class for loading datasets and generating prompts.
        
        Parameters:
        dataset_paths (dict): A dictionary with dataset names as keys and file paths as values.
        test_size (float): The proportion of the dataset to include in the test split.
        dev_size (float): The proportion of the training set to include in the dev split.
        prompt_type (str): The type of prompt to generate. (default: "reasoning")
        """
        self.dataset_paths = dataset_paths
        self.test_size = test_size
        self.dev_size = dev_size
        self.prompt_type = prompt_type
        self.datasets = {}

        # Load and split datasets
        self.load_and_split_datasets()

    def load_and_split_datasets(self):
        """Loads the datasets from files and splits them into train, dev, and test sets."""
        for dataset_name, dataset_path in self.dataset_paths.items():
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

            # Load the dataset from JSON file
            breakpoint()
            dataset = load_dataset("json", data_files=dataset_path)['train']

            # Split the dataset into train, test, and dev sets
            dataset = dataset.train_test_split(test_size=self.test_size)
            dataset['dev'] = dataset['train'].train_test_split(test_size=self.dev_size)['test']
            self.datasets[dataset_name] = dataset

    def generate_prompt(self, query):
        """
        Generate a prompt based on the prompt type.

        Parameters:
        query (str): The input query or question.

        Returns:
        str: The formatted prompt.
        """
        if self.prompt_type == "reasoning":
            return (f"Your task is to answer the question below. "
                    f"Give step-by-step reasoning before you answer, "
                    f"and when you’re ready to answer, please use the format `Final answer:...`.\n"
                    f"Question: {query}\nSolution: ")
        else:
            raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

    def get_train_data(self, dataset_name):
        """Returns the training dataset for the specified dataset."""
        return self.datasets[dataset_name]['train']

    def get_dev_data(self, dataset_name):
        """Returns the development (dev) dataset for the specified dataset."""
        return self.datasets[dataset_name]['dev']

    def get_test_data(self, dataset_name):
        """Returns the test dataset for the specified dataset."""
        return self.datasets[dataset_name]['test']