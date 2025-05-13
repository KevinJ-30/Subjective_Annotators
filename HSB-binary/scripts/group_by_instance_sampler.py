from torch.utils.data import Sampler
import numpy as np

class GroupByInstanceBatchSampler(Sampler):
    def __init__(self, dataset, max_batch_size=32, shuffle=True):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle

        # Build mapping from original_id to list of indices
        self.instance_to_indices = {}
        for idx, text_id in enumerate(dataset.text_ids):
            self.instance_to_indices.setdefault(text_id, []).append(idx)
        self.instance_ids = list(self.instance_to_indices.keys())

    def __iter__(self):
        instance_ids = self.instance_ids.copy()
        if self.shuffle:
            np.random.shuffle(instance_ids)
        batch = []
        for text_id in instance_ids:
            indices = self.instance_to_indices[text_id]
            if len(batch) + len(indices) > self.max_batch_size and batch:
                yield batch
                batch = []
            batch.extend(indices)
        if batch:
            yield batch

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.max_batch_size))