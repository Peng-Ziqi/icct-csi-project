import torch
import torch.nn as nn

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_flops(model, input_shape, device="cpu"):
    flops = {"total": 0}

    def conv_hook(m, inp, out):
        x = inp[0]
        batch = x.size(0)
        out_h, out_w = out.size(2), out.size(3)
        kernel_ops = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        ops = batch * m.out_channels * out_h * out_w * kernel_ops
        flops["total"] += ops

    def linear_hook(m, inp, out):
        x = inp[0]
        batch = x.size(0)
        ops = batch * m.in_features * m.out_features
        flops["total"] += ops

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape, device=device)
        model(dummy)

    for h in hooks:
        h.remove()

    # 返回 MACs 与 FLOPs(≈2*MACs)
    macs = flops["total"]
    flops2 = 2 * macs
    return macs, flops2
