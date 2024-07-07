import torch
import numpy as np
from gym import spaces

RT1_ACTION_SPACE = {
            'terminate': spaces.Discrete(2),
            'world_vector': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'rotation_delta': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            'gripper_closedness': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
}
RT1_ACTION_ORDER = ['terminate', 'world_vector', 'rotation_delta', 'gripper_closedness']


class ActionTokenizer:
    def __init__(self, action_space, action_bin_size, action_order=None):
        """
        初始化RT1ActionTokenizer类。
        参数:
        action_space: 动作空间的定义，一个包含所有动作及其属性的字典。
        vocab_size: 每个动作维度的bin数量，用于离散化动作。
        action_order: 动作的顺序，用于确定token化动作的顺序。

        >>> Example
        action_space = {
             'terminate': spaces.Discrete(2),
             'world_vector': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
             'rotation_delta': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
             'gripper_closedness': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
         }
        tokenizer = RT1ActionTokenizer(action_space, vocab_size, action_order=['terminate', 'world_vector', 'rotation_delta', 'gripper_closedness'])
        """
        self.action_space = action_space
        self.vocab_size = action_bin_size
        self.action_order = action_order if action_order is not None else list(action_space.keys())
        # 计算每个动作的token数量
        self.tokens_per_action = sum(len(action_space[action].shape) if action_space[action].dtype != torch.int32 else 1 for action in self.action_order)
    
    def tokenize(self, action):
        """
        接收一个动作字典action，然后遍历 action_order 中的每个动作，根据动作的规格（离散或连续）进行 token化。
        对于离散空间，直接使用动作值作为 token；对于连续空间，先进行裁剪和标准化，然后将其转换为整数token。
        """
        action_tokens = []
        for k in self.action_order:
            a = action[k]  # a is a tensor
            spec = self.action_space[k]

            if isinstance(spec, spaces.Discrete):
                # 对于离散空间，不需要裁剪，直接取值作为token
                token = a
            elif isinstance(spec, spaces.Box):
                # 将 spec.low 和 spec.high 转换为张量
                low_tensor = torch.tensor(spec.low, dtype=a.dtype)
                high_tensor = torch.tensor(spec.high, dtype=a.dtype)
                # 对于连续空间，需要裁剪并标准化
                a = torch.clamp(a, low_tensor, high_tensor)  # 使用low_tensor和high_tensor属性
                token = (a - low_tensor) / (high_tensor - low_tensor) * (self.vocab_size - 1)
                token = token.to(torch.int32)

            else:
                raise ValueError(f"Unsupported action space type: {type(spec)}")
            
            action_tokens.append(token)  # 增加维度
        # 将所有动作的token连接起来
        action_tokens = torch.cat(action_tokens, dim=-1)
        return action_tokens


    def detokenize(self, action_tokens):
        """
        对于action_tokens里的action_token，按照action_order进行detokenize
        """
        action = {}
        token_index = 0
        for k in self.action_order:
            spec = self.action_space[k]
            if isinstance(spec, spaces.Discrete):
                # terminate
                # Discrete 类型的动作空间，直接取索引作为动作值
                action_dim = spec.n  # 使用 spec.n 获取离散空间的大小
                token = action_tokens[..., token_index]  # 获取当前 token
                # 确保 token 在合法范围内
                token = torch.clamp(token, 0, action_dim - 1)
                action[k] = token
                token_index += 1

            elif isinstance(spec, spaces.Box):
                # world_vector, rotation_delta, gripper_closedness
                # Box 类型需要检查 shape 是否存在
                if spec.shape:  # 确保 spec.shape 不是空元组
                    action_dim = spec.shape[0]
                else:
                    raise ValueError("Box space has no shape defined")
                
                actions = []
                for _ in range(action_dim):
                    a = action_tokens[..., token_index:token_index + 1].float()
                    a = a / (self.vocab_size - 1)
                    a = a * (spec.high - spec.low) + spec.low
                    actions.append(a[0])
                    token_index += 1

                action[k] = actions

            else:
                raise ValueError(f"Unsupported action space type: {type(spec)}")
        return action


# # 示例用法：
# # 定义动作规格和词汇表大小
# action_space = {
#     'terminate': spaces.Discrete(2),
#     'world_vector': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
#     'rotation_delta': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
#     'gripper_closedness': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
# }

# vocab_size = 256

# # 实例化tokenizer
# tokenizer = RT1ActionTokenizer(action_space, vocab_size, action_order=['terminate', 'world_vector', 'rotation_delta', 'gripper_closedness'])

# # 示例动作张量
# action = {
#     'terminate': torch.tensor([0], dtype=torch.int32),
#     'world_vector': torch.tensor([0.9, 0.8, -0.3], dtype=torch.float32),
#     'rotation_delta': torch.tensor([-0.1, 0.2, 0.6], dtype=torch.float32),
#     'gripper_closedness': torch.tensor([0.9], dtype=torch.float32)
# }

# action_tokens = tokenizer.tokenize(action)
# action_detokenized = tokenizer.detokenize(action_tokens)