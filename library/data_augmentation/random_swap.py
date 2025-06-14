import random

class RandomSwap:
    def __init__(self, n=1, seed=42):

        """
        Initalizes the RandomSwap object.

        Args:
            n (int): The number of pairs of words to swap.
            seed (int): Random seed for reproducibility.
        """

        self.n = n
        self.seed = seed

    def run(self, text):
        
        """
        Randomly swap n pairs of words in the text.

        Args:
            text (str): The input text to be modified.

        Returns:
            str: The text after applying random swap.
        """

        random.seed(self.seed)  # Set random seed for reproducibility

        words = text.split()
        if len(words) <= 1:
            return text
        
        new_words = words.copy()
        for _ in range(self.n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)

# def random_swap(text, n=1, seed=42):
#     """
#     Randomly swap n pairs of words in the text.

#     Args:
#         text (str): The input text to be modified.
#         n (int): The number of pairs of words to swap.
#         seed (int): Random seed for reproducibility.

#     """

#     random.seed(seed)  # Set random seed for reproducibility

#     words = text.split()
#     if len(words) <= 1:
#         return text
    
#     new_words = words.copy()
#     for _ in range(n):
#         idx1, idx2 = random.sample(range(len(new_words)), 2)
#         new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    
#     return ' '.join(new_words)