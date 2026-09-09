import torch
import torch.nn as nn


class SequenceJSCCWrapper(nn.Module):
    """
    Wrap a frame-wise JSCC model for CSI sequences.

    Input:
        x_seq:  [B,T,2,32,256]
        snr_db: scalar, tensor [B], tensor [B,1], tensor [B,T], or tensor [B,T,1]

    Output:
        coarse_seq: [B,T,2,32,256]

    This wrapper does not modify the base JSCC model.
    It applies the original spatial JSCC backbone frame by frame.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def _format_snr_for_frame(self, snr_db, batch_size, frame_idx, device):
        if isinstance(snr_db, (float, int)):
            return float(snr_db)

        if not torch.is_tensor(snr_db):
            raise TypeError(f"Unsupported snr_db type: {type(snr_db)}")

        snr_db = snr_db.to(device)

        if snr_db.ndim == 0:
            return float(snr_db.item())

        if snr_db.ndim == 1:
            if snr_db.shape[0] == batch_size:
                return snr_db
            if snr_db.shape[0] == 1:
                return float(snr_db.item())

        if snr_db.ndim == 2:
            if snr_db.shape == (batch_size, 1):
                return snr_db
            if snr_db.shape[0] == batch_size:
                return snr_db[:, frame_idx]

        if snr_db.ndim == 3:
            if snr_db.shape[0] == batch_size:
                return snr_db[:, frame_idx, :]

        raise ValueError(f"Unsupported snr_db shape: {tuple(snr_db.shape)}")

    def _forward_frame(self, x_frame, snr_frame, gate_mode="estimate"):
        """
        Compatible with:
        1) SEJSCCSNRPrior:
           y, snr_est, ratio, k = model(x, snr_db=..., gate_mode=...)
        2) Other JSCC variants:
           y = model(x, snr_db)
        """
        try:
            out = self.base_model(x_frame, snr_db=snr_frame, gate_mode=gate_mode)
        except TypeError:
            out = self.base_model(x_frame, snr_frame)

        if torch.is_tensor(out):
            y = out
            aux = None
        elif isinstance(out, (tuple, list)):
            y = out[0]
            aux = out[1:]
        elif isinstance(out, dict):
            if "y" in out:
                y = out["y"]
            elif "x_hat" in out:
                y = out["x_hat"]
            elif "coarse" in out:
                y = out["coarse"]
            else:
                raise KeyError("Dict output does not contain y/x_hat/coarse.")
            aux = out
        else:
            raise TypeError(f"Unsupported base model output type: {type(out)}")

        return y, aux

    def forward(self, x_seq, snr_db, gate_mode="estimate", return_aux=False):
        if x_seq.ndim != 5:
            raise ValueError(f"Expected x_seq [B,T,2,32,256], got {tuple(x_seq.shape)}")

        batch_size, seq_len, channels, height, width = x_seq.shape
        if channels != 2 or height != 32 or width != 256:
            raise ValueError(f"Unexpected CSI shape: {tuple(x_seq.shape)}")

        outputs = []
        aux_list = []

        for ti in range(seq_len):
            x_frame = x_seq[:, ti]
            snr_frame = self._format_snr_for_frame(
                snr_db=snr_db,
                batch_size=batch_size,
                frame_idx=ti,
                device=x_seq.device,
            )

            y_frame, aux = self._forward_frame(
                x_frame=x_frame,
                snr_frame=snr_frame,
                gate_mode=gate_mode,
            )

            outputs.append(y_frame)
            aux_list.append(aux)

        coarse_seq = torch.stack(outputs, dim=1)

        if not return_aux:
            return coarse_seq

        return {
            "coarse": coarse_seq,
            "aux": aux_list,
        }
