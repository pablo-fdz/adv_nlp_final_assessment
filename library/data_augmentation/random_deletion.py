import random

class RandomDeletion:
    def __init__(self, p=0.1, seed=42):

        """
        Initalizes the RandomDeletion object.

        Args:
            text (str): The input text from which words will be deleted.
            p (float): Probability of deleting each word. Default is 0.1.
            seed (int): Random seed for reproducibility.
        """

        self.p = p
        self.seed = seed

    def run(self, text):
        
        """
        Applies random deletion to the text.

        Args:
            text (str): The input text to be modified.

        Returns:
            str: The text after applying random deletion.
        """

        random.seed(self.seed)  # Set the random seed for reproducibility

        words = text.split()
        if len(words) <= 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > self.p:
                new_words.append(word)  # Append the word with probability (1 - p)
        
        if len(new_words) == 0:  # If all words were deleted, keep a random word
            return random.choice(words)
        
        return ' '.join(new_words)

# def random_deletion(text, p=0.1, seed=42):
#     """
#     Randomly delete words from the text with probability p.

#     Args:
#         text (str): The input text from which words will be deleted.
#         p (float): Probability of deleting each word. Default is 0.1.
#         seed (int): Random seed for reproducibility.
#     """
#     random.seed(seed)  # Set the random seed for reproducibility

#     words = text.split()
#     if len(words) <= 1:
#         return text
    
#     new_words = []
#     for word in words:
#         if random.random() > p:
#             new_words.append(word)  # Append the word with probability (1 - p)
    
#     if len(new_words) == 0:  # If all words were deleted, keep a random word
#         return random.choice(words)
    
#     return ' '.join(new_words)