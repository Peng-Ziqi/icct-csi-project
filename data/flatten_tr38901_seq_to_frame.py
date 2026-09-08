import os
import argparse
import torch


def load_seq_obj(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    if "x" not in obj:
        raise KeyError(f"Missing key 'x' in {path}")
    return obj


def flatten_split(seq_path, out_path):
    obj = load_seq_obj(seq_path)

    x_seq = obj["x"]
    if x_seq.ndim != 5:
        raise ValueError(f"Expected x shape [N,T,2,32,256], got {tuple(x_seq.shape)}")

    n, t, c, h, w = x_seq.shape
    if c != 2 or h != 32 or w != 256:
        raise ValueError(f"Unexpected CSI shape {tuple(x_seq.shape)}, expected [N,T,2,32,256]")

    x_frame = x_seq.reshape(n * t, c, h, w).contiguous()

    out_obj = {
        "x": x_frame,
        "source_seq_path": seq_path,
        "original_shape": tuple(x_seq.shape),
        "flattened_shape": tuple(x_frame.shape),
    }

    if "speed_kmh" in obj:
        speed = obj["speed_kmh"]
        out_obj["speed_kmh"] = speed.repeat_interleave(t)

    if "scenario_id" in obj:
        scenario = obj["scenario_id"]
        out_obj["scenario_id"] = scenario.repeat_interleave(t)

    torch.save(out_obj, out_path)

    print(f"[Saved] {out_path}")
    print(f"  original:  {tuple(x_seq.shape)}")
    print(f"  flattened: {tuple(x_frame.shape)}")
    if "speed_kmh" in out_obj:
        print(f"  speeds:    {torch.unique(out_obj['speed_kmh'])}")
    if "scenario_id" in out_obj:
        print(f"  scenarios: {torch.unique(out_obj['scenario_id'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    proc_dir = os.path.join(args.data_root, "processed")

    splits = ["train", "val", "test"]

    for split in splits:
        seq_path = os.path.join(proc_dir, f"{split}_tr38901_seq.pt")
        out_path = os.path.join(proc_dir, f"{split}_tr38901_frame.pt")

        if not os.path.exists(seq_path):
            raise FileNotFoundError(seq_path)

        flatten_split(seq_path, out_path)

    print("\n[Done] Frame-wise TR 38.901 dataset generated.")


if __name__ == "__main__":
    main()
