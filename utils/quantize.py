# utils/quantize.py
import torch
import copy
import torch.nn as nn


def _uniform_sym_quant_per_tensor(w: torch.Tensor, bits: int):
    if bits >= 32:
        return w
    assert bits in [2, 4, 8]
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().max()
    if max_abs == 0:
        return w.clone()
    scale = max_abs / qmax
    w_int = torch.round(w / scale).clamp(-qmax, qmax)
    return w_int * scale


def _uniform_sym_quant_per_channel_out(w: torch.Tensor, bits: int):
    """
    per-channel 对输出通道量化:
    - Linear: [out, in]
    - Conv2d: [out, in, k, k]
    """
    if bits >= 32:
        return w
    assert bits in [2, 4, 8]
    qmax = 2 ** (bits - 1) - 1

    wq = w.clone()
    out_ch = w.shape[0]
    for c in range(out_ch):
        wc = w[c]
        max_abs = wc.abs().max()
        if max_abs == 0:
            continue
        scale = max_abs / qmax
        wc_int = torch.round(wc / scale).clamp(-qmax, qmax)
        wq[c] = wc_int * scale
    return wq


def quantize_model_copy(model, rules: dict, keep_bias_fp32=True):
    """
    rules: {name_substr: bits}
    若命中参数:
      - weight: Linear/Conv使用per-channel；其他用per-tensor
      - bias: 默认不量化(keep_bias_fp32=True)
    """
    qmodel = copy.deepcopy(model)

    module_map = dict(qmodel.named_modules())

    with torch.no_grad():
        for n, p in qmodel.named_parameters():
            hit_bits = None
            for key, bits in rules.items():
                if key in n:
                    hit_bits = bits
                    break
            if hit_bits is None:
                continue

            # bias保持FP32
            if keep_bias_fp32 and n.endswith(".bias"):
                continue

            # 找到所属module名字
            mod_name = n.rsplit(".", 1)[0] if "." in n else ""
            mod = module_map.get(mod_name, None)

            if n.endswith(".weight") and isinstance(mod, (nn.Linear, nn.Conv2d)):
                p.copy_(_uniform_sym_quant_per_channel_out(p, hit_bits))
            else:
                p.copy_(_uniform_sym_quant_per_tensor(p, hit_bits))

    return qmodel


def model_size_bits(model, default_bits=32, rules=None, keep_bias_fp32=True):
    if rules is None:
        rules = {}
    total = 0
    for n, p in model.named_parameters():
        bits = default_bits
        for key, b in rules.items():
            if key in n:
                bits = b
                break
        if keep_bias_fp32 and n.endswith(".bias"):
            bits = 32
        total += p.numel() * bits
    return total
