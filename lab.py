import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
# import torch 
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


# load a single dataset
dataset_kwargs = make_oxe_dataset_kwargs(
    "cmu_stretch",
    DATA_PATH,
)

# TensorFlow loader
dataset = make_single_dataset(dataset_kwargs, train=True) # load the train split
iterator = dataset.iterator()
traj = next(iterator)
print("Top-level keys: ", traj.keys())                  # dict_keys(['observation', 'task', 'action', 'dataset_name', 'action_pad_mask'])
print("Observation keys: ", traj["observation"].keys()) # dict_keys(['image_primary', 'timestep', 'pad_mask_dict', 'timestep_pad_mask', 'task_completed'])
print("Task keys: ", traj["task"].keys())               # dict_keys(['language_instruction', 'pad_mask_dict'])

images = traj["observation"]["image_primary"]
instruction = traj["task"]["language_instruction"]
print(images.shape)         # should be: (traj_len, window_size, height, width, channels), (window_size defaults to 1), (196, 1, 128, 128, 3)
print(instruction.shape)    # should be: (traj_len,)
print(instruction[0]) # "pull open a dishwasher"


# RT-1 inference
# frames = torch.randn(2, 5, 3, 300,300)
# instruction = ["pick", "place"]

# rt1 = RT1(seq_len=5)

# logits, tokens = rt1(frames, instruction)
# print(logits.shape)
# print(tokens)
