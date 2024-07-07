import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from .embedding import CLIPEmbedding
from .efficientnet_pytorch import EfficientNet
from .transformer import Transformer
from .token_learner import TokenLearner
# from .action_tokenize import ActionTokenizer, RT1_ACTION_SPACE, RT1_ACTION_ORDER


class RT1(BaseModel):
    """
    Robot Transformer-1
    """
    def __init__(self, seq_len, action_token_len=8, action_bin_size=256):
        super().__init__()

        self.action_token_size = action_token_len
        self.action_bin_size = action_bin_size

        # Instruction embedding using clip ViT-B/32 encoder which gives 1 x 512 sentence embedding
        self.text_embedding = CLIPEmbedding()

        # FiLM EfficientNet B3, image encoder conditioned on the instruction embedding
        self.film_efficientnetB3 = EfficientNet.from_pretrained_film("efficientnet-b3", weights_path='checkpoints/b3.pth')
        # 1x1 conv to resize the channel
        self.conv1x1 = nn.Conv2d(
            in_channels= 1536,
            out_channels= 512,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # token leaner
        self.token_learner = TokenLearner(num_tokens=action_token_len, input_dim=512)

        # transformer
        self.transformer = Transformer(token_len=seq_len*action_token_len, num_layer=8, action_bin_size=action_bin_size)

        # action tokenizer
        # self.action_tokenizer = ActionTokenizer(action_space=RT1_ACTION_SPACE, 
        #                                         action_bin_size=action_bin_size, 
        #                                         action_order=RT1_ACTION_ORDER)



    def forward(self, frames, instructions):
        """
        Args:
            frames: bs x t x 3 x 300 x 300
            instruction: bs x 1
        """
        bs, t, c, h, w = frames.shape
        frames = frames.reshape(bs*t, c, h, w)

        # embed instruction
        instruction_embedding = self.text_embedding(instructions) # bs x embed_dim
        instruction_embedding = instruction_embedding.unsqueeze(1).expand(-1, t, -1) # bs x t x embed_dim
        instruction_embedding = instruction_embedding.reshape(bs * t, -1) # bs * t x embed_dim

        # process each frame, get instruction conditioned visual feature
        tokens = self.film_efficientnetB3.extract_features(frames, instruction_embedding)
        tokens = self.conv1x1(tokens) # torch.Size([6, 512, 10, 10])

        # pass through token learner
        tokens = self.token_learner(tokens) # bs*t x 8 x 512
        tokens = tokens.view(bs, self.action_token_size * t, 512) # bs x t*8 x 512

        # pass through transformer
        logits = self.transformer(tokens) # bs x action_size*t x 256
        bs = logits.shape[0]
        logits = logits.view(bs, self.action_token_size, t, self.action_bin_size)
        logits = torch.mean(logits, dim=2) # bs x 8 x 256

        # logits to probilities and discrete actions tokens
        probs = F.softmax(logits, dim=-1)
        action_tokens = torch.argmax(probs, dim=-1) # bs x 8

        return logits, action_tokens
        