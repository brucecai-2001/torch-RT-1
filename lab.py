import torch 
import tensorflow as tf
from torch.nn import functional as F
from torch.utils.data import DataLoader

#from model import RT1
import tqdm
from data_loader.octo.data.dataset import make_interleaved_dataset, make_single_dataset
from data_loader.octo.data.oxe import make_oxe_dataset_kwargs_and_weights, make_oxe_dataset_kwargs
from data_loader.data_loaders import TorchRLDSDataset

from PIL import Image

DATA_PATH = "/root/autodl-tmp/cmu_stretch"

tf.config.set_visible_devices([], "GPU")


# LOAD A SINGLE DATASET
dataset_kwargs = make_oxe_dataset_kwargs(
    "cmu_stretch",
    DATA_PATH,
)

dataset = make_single_dataset(dataset_kwargs, train=True) # load the train split

iterator = dataset.iterator()

traj = next(iterator)
print("Top-level keys: ", traj.keys())
print("Observation keys: ", traj["observation"].keys())
print("Task keys: ", traj["task"].keys())

images = traj["observation"]["image_primary"]
print(images.shape)  # should be: (traj_len, window_size, height, width, channels), (window_size defaults to 1)

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
# print("make pytorch_dataset")

# dataloader = DataLoader(
#     pytorch_dataset,
#     batch_size=2,
#     num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
# )
# print("make dataloader")

# for i, sample in tqdm.tqdm(enumerate(dataloader)):
#     print(sample)
#     if i == 8:
#         break
# print("dataloder finishes")


# frames = torch.randn(2, 5, 3, 300,300)
# instruction = ["pick", "place"]

# rt1 = RT1(seq_len=5)

# logits, tokens = rt1(frames, instruction)
# print(logits.shape)
# print(tokens)
