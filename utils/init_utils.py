# utils/init_utils.py
import torch.nn as nn

def _init_conv_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def _init_linear_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def _init_bn(m):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_model(model, scheme="default"):
    """
    scheme:
      - default: 不改动
      - he_all: Conv + Linear 全部He
      - he_conv_only: 仅Conv He
      - he_conv_mainfc: Conv He + encoder_fc/decoder_fc He，不改SE内fc
    """
    if scheme == "default":
        return model

    if scheme == "he_all":
        model.apply(_init_conv_he)
        model.apply(_init_linear_he)
        model.apply(_init_bn)
        return model

    if scheme == "he_conv_only":
        model.apply(_init_conv_he)
        model.apply(_init_bn)
        return model

    if scheme == "he_conv_mainfc":
        model.apply(_init_conv_he)
        model.apply(_init_bn)
        # 只初始化主干两个FC
        if hasattr(model, "encoder_fc"):
            nn.init.kaiming_normal_(model.encoder_fc.weight, mode='fan_in', nonlinearity='relu')
            if model.encoder_fc.bias is not None:
                nn.init.zeros_(model.encoder_fc.bias)
        if hasattr(model, "decoder_fc"):
            nn.init.kaiming_normal_(model.decoder_fc.weight, mode='fan_in', nonlinearity='relu')
            if model.decoder_fc.bias is not None:
                nn.init.zeros_(model.decoder_fc.bias)
        return model

    raise ValueError(f"Unknown init scheme: {scheme}")
