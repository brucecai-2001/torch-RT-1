import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import tqdm
from data_loader.octo.data.dataset import make_interleaved_dataset
from data_loader.octo.data.oxe import make_oxe_dataset_kwargs_and_weights

# DATA_PATH = "/Users/caixinyu/Downloads/cmu_stretch"

# tf.config.set_visible_devices([], "CPU")


class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)


# dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
#     "oxe_magic_soup",
#     DATA_PATH,
#     load_camera_views=("primary", "wrist"),
# )
# print("make_oxe_dataset_kwargs_and_weights")

# dataset = make_interleaved_dataset(
#     dataset_kwargs_list,
#     sample_weights,
#     train=True,
#     shuffle_buffer_size=1000,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
#     batch_size=None,  # batching will be handles in PyTorch Dataloader object
#     balance_weights=True,
#     traj_transform_kwargs=dict(
#         goal_relabeling_strategy="uniform",
#         window_size=2,
#         action_horizon=4,
#         subsample_length=100,
#     ),
#     frame_transform_kwargs=dict(
#         image_augment_kwargs={
#             "primary": dict(
#                 random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
#                 random_brightness=[0.1],
#                 random_contrast=[0.9, 1.1],
#                 random_saturation=[0.9, 1.1],
#                 random_hue=[0.05],
#                 augment_order=[
#                     "random_resized_crop",
#                     "random_brightness",
#                     "random_contrast",
#                     "random_saturation",
#                     "random_hue",
#                 ],
#             ),
#             "wrist": dict(
#                 random_brightness=[0.1],
#                 random_contrast=[0.9, 1.1],
#                 random_saturation=[0.9, 1.1],
#                 random_hue=[0.05],
#                 augment_order=[
#                     "random_brightness",
#                     "random_contrast",
#                     "random_saturation",
#                     "random_hue",
#                 ],
#             ),
#         },
#         resize_size=dict(
#             primary=(256, 256),
#             wrist=(128, 128),
#         ),
#         num_parallel_calls=200,
#     ),
#     traj_transform_threads=2,
#     traj_read_threads=2,
# )
# print("make_interleaved_dataset")


# pytorch_dataset = TorchRLDSDataset(dataset)
# print("pytorch_dataset")

# dataloader = DataLoader(
#     pytorch_dataset,
#     batch_size=2,
#     num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
# )
# print("dataloader")

# for i, sample in tqdm.tqdm(enumerate(dataloader)):
#     print("sample{i}")
#     if i == 10:
#         break
