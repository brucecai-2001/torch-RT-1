import clip
import torch.nn as nn

class CLIPEmbedding(nn.Module):
    def __init__(self):
        super(CLIPEmbedding, self).__init__()
        """
        :param model_name: 预训练模型的名称。
        """
        self.model,_ = clip.load("ViT-B/32") 

    def forward(self, sentence):
        """
        对一个句子列表进行编码，生成句子嵌入。
        :param sentences: 要编码的句子。
        :return: 句子embedding 1 x 512
        """
        tokenized = clip.tokenize(sentence, context_length=77)
        text_features = self.model.encode_text(tokenized)
        return text_features
    

# c = CLIPEmbedding()
# instruction = ["pick", "place"]
# instruction = c(instruction).unsqueeze(1).expand(-1, 3, -1)
# print(instruction.shape)