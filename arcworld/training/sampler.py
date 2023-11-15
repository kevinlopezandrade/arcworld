import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List

from torch.utils.data import Sampler

from arcworld.internal.constants import Task
from arcworld.training.dataloader import ARCDataset


class ARCBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: ARCDataset,
        max_batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """
        This is a custom batch sampler for the ARC challanged targeted for the
        Sformer. It ensure that every batch has the same number of input-output
        pairs even if the dataset contains samples with varying input-output
        pairs.

        Two levels of shuffling exist: The first level works within a subgroup
        of a fixed number of input-output pairs. The number of batches of this
        first subgroup its the same accross epochs. If shuffle is True, then
        the B1 of this subgroup should contain differente indices if the
        iterator is started again, and the same for every BN and every
        subgroup. The second of level of shuffling happens at the top level.
        Once every subgroup has been batched we collect a list of batches,
        which we shuffle as well.

        Args:
            dataset: The ARCDataset for which to create the sampler.
            max_batch_size: Prefered max_batch_size. Note the this is an upper
                bound.
            shuffle: If to shuffle the batches after every call to __iter__.
            drop_last: If drop the batches that do not contain up to
                max_batch_size samples.
        """
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._groups = ARCBatchSampler.group_tasks_by_length(self.dataset.data)
        self._group_indices = [indices for indices in self._groups.values()]

        # Only used if no there is no shuffling.
        self._batches = ARCBatchSampler._create_batches(
            self._group_indices, self.max_batch_size, self.drop_last
        )

    @staticmethod
    def group_tasks_by_length(data: List[Task]) -> Dict[int, List[int]]:
        groups: defaultdict[int, list[int]] = defaultdict(list[int])
        for i, task in enumerate(data):
            n = len(task)
            groups[n].append(i)

        return groups

    @staticmethod
    def _create_batches(
        group_indices: List[List[int]], max_batch_size: int, drop_last: bool = False
    ) -> List[List[int]]:
        batches: List[List[int]] = []

        for group in group_indices:
            num_batches = math.ceil(len(group) / max_batch_size)
            for i in range(num_batches):
                batch = group[i * max_batch_size : (i + 1) * max_batch_size]

                if drop_last:
                    if len(batch) == max_batch_size:
                        batches.append(batch)
                else:
                    batches.append(batch)

        return batches

    @staticmethod
    def _crated_shuffled_batches(
        group_indices: List[List[int]], max_batch_size: int, drop_last: bool = False
    ):
        # Shuffle at the first level.
        for indices in group_indices:
            random.shuffle(indices)

        batches = ARCBatchSampler._create_batches(
            group_indices, max_batch_size, drop_last
        )
        # Shuffle at the second level.
        random.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffling once the indices are computed to not create copies
        # or modify in place the List[Task]
        if self.shuffle:
            batches = ARCBatchSampler._crated_shuffled_batches(
                self._group_indices, self.max_batch_size, self.drop_last
            )
        else:
            batches = self._batches

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)
