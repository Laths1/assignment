"""
This class generates train, val and test set according to a specified split.
There is no data leak, the training, validation and test sets are all independent of each other.
To ensure that the entire dataset is equal to the training_batch, make sure that training_batch is divisible by the sum of the splits.
"""
import wcst
import numpy as np

class Dataset_Loader:
    def __init__(self, training_batch, classification_batch, train_split, val_split, test_split, context_switch_interval):
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.classification_batch = classification_batch
        self.training_batch = training_batch
        self.context_switch_interval = context_switch_interval

    def _trial_key(self, input_batch, target_batch):
        """
        Create a unique key per trial including:
        - category cards
        - example card
        - example label
        - question card
        - question label (target)
        """
        keys = []
        for i in range(input_batch.shape[0]):
            cat_cards = tuple(input_batch[i, :4])
            example_card = input_batch[i, 4]
            example_label = input_batch[i, 5]
            question_card = input_batch[i, 6]
            question_label = target_batch[i, 0]
            keys.append((cat_cards,
                         example_card,
                         example_label,
                         question_card,
                         question_label))
        return keys

    def load_data(self):
        """
        The sum of train, val and test splits should equal training_batch.
        Ensures no duplicate trials across train/val/test.
        """
        wcst_env = wcst.WCST(self.classification_batch)
        train_data = []
        val_data = []
        test_data = []
        seen = set()  # track all seen trial keys

        n_train = int(np.floor(self.training_batch * self.train_split))
        n_val = int(np.floor(self.training_batch * self.val_split))
        n_test = int(np.floor(self.training_batch * self.test_split))

        def fill_dataset(n_items, dataset_list, start_count=0):
            count = 0
            while count < n_items:
                if (start_count + count) % self.context_switch_interval == 0 and (start_count + count) != 0:
                    wcst_env.context_switch()

                input_batch, target_batch = next(wcst_env.gen_batch())
                keys = self._trial_key(input_batch, target_batch)

                new_indices = [i for i, k in enumerate(keys) if k not in seen]
                if len(new_indices) == 0:
                    continue  # skip this batch 

                for i in new_indices:
                    seen.add(keys[i])
                    dataset_list.append((input_batch[i:i+1], target_batch[i:i+1]))
                    count += 1
                    if count >= n_items:
                        break
            return count

        # fill each split ensuring uniqueness
        c1 = fill_dataset(n_train, train_data, start_count=0)
        c2 = fill_dataset(n_val, val_data, start_count=c1)
        c3 = fill_dataset(n_test, test_data, start_count=c1+c2)

        return train_data, val_data, test_data

# Example usage:
loader = Dataset_Loader(training_batch=10_000, classification_batch=8, train_split=0.7, val_split=0.15, test_split=0.15, context_switch_interval=50)
train_data, val_data, test_data = loader.load_data()
print("="*20)
print("Dataset Loaded")
print("="*20)
print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}, Test data size: {len(test_data)}")
print(f"Training data example: {train_data[0]}")  
print(f"Validation data example: {val_data[0]}")  
print(f"Test data example: {test_data[0]}")  

# check if there are any overlaps
print("="*20)
print("Checking for overlaps between datasets...")
print("="*20)
def dataset_to_keys(loader, dataset):
    """Convert a list of (input_batch, target_batch) into a set of trial keys."""
    keys = set()
    for (input_row, target_row) in dataset:
        k = loader._trial_key(input_row, target_row)[0]
        keys.add(k)
    return keys

train_keys = dataset_to_keys(loader, train_data)
val_keys = dataset_to_keys(loader, val_data)
test_keys = dataset_to_keys(loader, test_data)

print("Train set size:", len(train_keys))
print("Val set size:", len(val_keys))
print("Test set size:", len(test_keys))
print("Train-Val Overlap:", len(train_keys.intersection(val_keys)))
print("Train-Test Overlap:", len(train_keys.intersection(test_keys)))
print("Val-Test Overlap:", len(val_keys.intersection(test_keys)))
