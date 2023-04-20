import torch
import numpy as np

from renaanalysis.learning.HT import Attention


def rollout(depth, attentions, discard_ratio, head_fusion, token_shape):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for i, attention in enumerate(attentions):
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-2), attention_heads_fused.size(-1))  # TODO check why this is not square for ExpertAttentionModel
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
            if i == depth:
                break

    # Look at the total attention between the class token,
    # and the tokens
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    mask = mask.reshape(token_shape).numpy()
    mask = mask / np.max(mask)
    return mask


class VITAttentionRollout:
    def __init__(self, model, device, attention_layer_class, token_shape, head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.device = device
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.token_shape = token_shape

        self.attention_layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, attention_layer_class):  # HT's attention layer
                module.register_forward_hook(self.get_attention)
                self.attention_layer_count += 1
        if self.attention_layer_count == 0:
            raise ValueError("No attention layer in the given model")
        if self.attention_layer_count != self.model.depth:
            raise ValueError(f"Model depth ({self.model.depth}) does not match attention layer count {self.attention_layer_count}")
        self.attentions = []

    def get_attention(self, module, input, output):
        attention_output = output
        self.attentions.append(attention_output[1].cpu())

    def __call__(self, depth, input_tensor, fix_sequence=None):
        if depth > self.attention_layer_count:
            raise ValueError(f"Given depth ({depth}) is greater than the number of attenion layers in the model ({self.attention_layer_count})")
        self.attentions = []

        output = self.model(input_tensor.to(self.device))

        return rollout(depth, self.attentions, self.discard_ratio, self.head_fusion, token_shape=self.token_shape)