# Robot Transformer 1
https://robotics-transformer1.github.io/
https://arxiv.org/pdf/2212.06817
https://fy8l8tj1wd.feishu.cn/docx/D56AdcUmvoCybTxNZO4c50nhnHh

*FOR STUDY AND PRACTICE PURPOSE ONLY*  
Implement the RT-1 in pytorch. Replace the text encoder into the text encoder of CLIP. Other parts keep unchanged.
```python
# RT-1 inference
frames = torch.randn(2, 5, 3, 300,300)
instruction = ["pick", "place"]

rt1 = RT1(seq_len=5)

logits, tokens = rt1(frames, instruction)
print(logits.shape)
print(tokens)

```
Test the Open X embodiment dataset loader. 

Only spport a simple dataloader example and a inference example. See lab.py for more details.

# Citations
Reference to these repos, appreciate a lot  
https://github.com/octo-models/octo  
https://github.com/google-research/robotics_transformer  
https://github.com/lucidrains/robotic-transformer-pytorch  