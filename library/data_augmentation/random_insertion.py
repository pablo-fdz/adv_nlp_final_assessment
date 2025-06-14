import random

class RandomInsertion:
    def __init__(self, ext_vocab=None, n=5, use_ext_prob=0.5, seed=42):

        """
        Initalizes the RandomInsertion object.

        Args:
            ext_vocab (list): Optional external vocabulary list
            n (int): Number of words to insert
            use_ext_prob (float): Probability of using external vocabulary
            seed (int): Random seed for reproducibility
        """

        self.ext_vocab = ext_vocab
        self.n = n
        self.use_ext_prob = use_ext_prob
        self.seed = seed

    def run(self, text):
        
        """
        Insert random words from text or external vocabulary.

        Args:
            text (str): The input text to be modified.

        Returns:
            str: The text after applying random insertion.
        """

        # Set the random seed for reproducibility
        random.seed(self.seed)

        words = text.split()
        if len(words) <= 1:
            return text
        
        new_words = words.copy()
        added_words = []
        
        for _ in range(self.n):
            # Decide whether to use external vocabulary
            if self.ext_vocab and random.random() < self.use_ext_prob:
                add_word = random.choice(self.ext_vocab)
            else:
                add_word = random.choice(words)
                
            added_words.append(add_word)
            pos = random.randint(0, len(new_words) - 1)
            new_words.insert(pos, add_word)
        
        return ' '.join(new_words)

# def random_insertion(text, ext_vocab=None, n=5, use_ext_prob=0.5, seed=42):
#     """Insert random words from text or external vocabulary. Set a random seed
#     for reproducibility in the code above (with `random.seed(seed)`).
    
#     Args:
#         text (str): Input text
#         ext_vocab (list): Optional external vocabulary list
#         n (int): Number of words to insert
#         use_ext_prob (float): Probability of using external vocabulary
#         seed (int): Random seed for reproducibility
#     """
#     # Set the random seed for reproducibility
#     random.seed(seed)

#     words = text.split()
#     if len(words) <= 1:
#         return text
    
#     new_words = words.copy()
#     added_words = []
    
#     for _ in range(n):
#         # Decide whether to use external vocabulary
#         if ext_vocab and random.random() < use_ext_prob:
#             add_word = random.choice(ext_vocab)
#         else:
#             add_word = random.choice(words)
            
#         added_words.append(add_word)
#         pos = random.randint(0, len(new_words) - 1)
#         new_words.insert(pos, add_word)
    
#     return added_words, ' '.join(new_words)