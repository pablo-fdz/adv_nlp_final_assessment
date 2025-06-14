from datasets import Dataset
import random

def augment_dataset(dataset: Dataset, techniques: list, augmentation_rate: float = 0.2, seed: int = 42):
    """
    Augment the dataset using specified techniques.
    
    Args:
        dataset (Dataset): The input dataset to augment. Must contain a 'text' and
            'label' field.
        techniques (list): List of augmentation techniques to apply.
        augmentation_rate (float): Rate of augmentations to apply per each augmentation technique,
            expressed as a proportion of the dataset size (e.g., 0.2 for 20%).
            This determines how many augmented examples will be generated.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Dataset: The augmented dataset.
    """
    # Initialize a seed counter for reproducibility
    seed_counter = 0
    
    # Calculate the number of augmentations based on the proportion
    n_augmentations = int(len(dataset) * augmentation_rate)
    
    # Start with all the original examples
    all_examples = list(dataset)
    
    # Ensure we have both 'text' and 'label' fields
    if 'text' not in dataset.features or 'label' not in dataset.features:
        raise ValueError("Dataset must contain 'text' and 'label' fields")
    
    # For each augmentation technique
    for technique in techniques:
        # Increment the seed counter for each augmentation
        seed_counter += 1
        random.seed(seed + seed_counter)

        # Create a list of indices and shuffle it to sample without replacement
        available_indices = list(range(len(dataset)))
        random.shuffle(available_indices)
        
        # Make sure we don't try to sample more indices than the dataset size
        actual_n_augmentations = min(n_augmentations, len(dataset))
        
        # Take the first n indices from the shuffled list
        selected_indices = available_indices[:actual_n_augmentations]
        
        # Generate n_augmentations for this technique
        for idx in selected_indices:
            original_example = dataset[idx]
            
            # Apply the augmentation technique
            augmented_text = technique.run(original_example['text'])
            
            # Create a new example with the same label
            augmented_example = {
                'text': augmented_text,
                'label': original_example['label']
            }
            
            # Add directly to our collection of all examples
            all_examples.append(augmented_example)
    
    # Create a new dataset with all examples (original + augmented)
    combined_dataset = Dataset.from_list(all_examples, features=dataset.features)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Augmented dataset size: {len(combined_dataset)}")
    print(f"Number of examples added: {len(combined_dataset) - len(dataset)}")
    
    return combined_dataset