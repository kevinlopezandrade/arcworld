import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

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
        Sformer. It ensures that every batch has the same number of input-output
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
        self.tasks = dataset.data
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._groups = ARCBatchSampler.group_tasks_by_length(self.tasks)
        self._group_indices = [indices for indices in self._groups.values()]

        # Only used if no there is no shuffling.
        self._batches = ARCBatchSampler.create_batches(
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
    def create_batches(
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
    def create_shuffled_batches(
        group_indices: List[List[int]], max_batch_size: int, drop_last: bool = False
    ):
        # Shuffle at the first level.
        for indices in group_indices:
            random.shuffle(indices)

        batches = ARCBatchSampler.create_batches(
            group_indices, max_batch_size, drop_last
        )
        # Shuffle at the second level.
        random.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffling once the indices are computed to not create copies
        # or modify in place the List[Task]
        if self.shuffle:
            batches = ARCBatchSampler.create_shuffled_batches(
                self._group_indices, self.max_batch_size, self.drop_last
            )
        else:
            batches = self._batches

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)


class ARCDistributedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: ARCDataset,
        num_replicas: int,
        rank: int,
        max_batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        """
        Distributed version of the ARCBatchSampler. Check the ARCBatchSampler for more
        information. The arguments follow the same definitions as in DistributedSampler
        from PyTorch.
        """
        self.tasks = dataset.data
        self.num_replicas = num_replicas
        self.rank = rank
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_samples_per_rank = math.ceil(len(self.tasks) / num_replicas)
        self.epoch = 0

        indices = list(range(len(self.tasks)))
        self.subset_tasks = [
            (index, self.tasks[index])
            for index in indices[
                self.rank
                * self.num_samples_per_rank : (self.rank + 1)
                * self.num_samples_per_rank
            ]
        ]

        groups = ARCDistributedBatchSampler.group_tasks_by_length(self.subset_tasks)
        self.grouped_indices = [indices for indices in groups.values()]
        self.batches = ARCBatchSampler.create_batches(
            self.grouped_indices, self.max_batch_size, self.drop_last
        )

    @staticmethod
    def group_tasks_by_length(
        data_indexed: List[Tuple[int, Task]]
    ) -> Dict[int, List[int]]:
        groups: defaultdict[int, list[int]] = defaultdict(list[int])
        for index, task in data_indexed:
            n = len(task)
            groups[n].append(index)

        return groups

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            batches = ARCBatchSampler.create_shuffled_batches(
                self.grouped_indices, self.max_batch_size, self.drop_last
            )
        else:
            batches = self.batches

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        """
        Ignored for the moment.
        """
        self.epoch = epoch
