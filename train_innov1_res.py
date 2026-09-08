# train_innov1_res.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet_res import SECsiNetRes
from utils.metrics import nmse
from utils.hooks_stats import ActivationStatsHook

from utils.logger import setup_log_redirect


def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        if "x" in obj: x = obj["x"]
        elif "data" in obj: x = obj["data"]
        elif "H" in obj: x = obj["H"]
        else: raise KeyError(f"{path} dict无x/data/H键")
    elif isinstance(obj, (list, tuple)):
        x = obj[0]
    else:
        raise TypeError(f"{path}类型不支持")
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--res-encoder-conv", action="store_true")
    parser.add_argument("--res-encoder-se", action="store_true")
    parser.add_argument("--res-refine-se", action="store_true")

    parser.add_argument("--collect-stats", action="store_true")
    parser.add_argument("--stats-batches", type=int, default=20)
    parser.add_argument("--stats-dir", type=str, default="stats")

    parser.add_argument("--log-file", type=str, default="")


    args = parser.parse_args()
    if args.log_file == "":
    # 自动命名，保持你之前风格
        tag = f"train_innov1_res_cr{args.cr}_rec{int(args.res_encoder_conv)}_res{int(args.res_encoder_se)}_rrs{int(args.res_refine_se)}.log"
        args.log_file = os.path.join("logs", tag)

    _tee = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_x = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    val_x = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))

    train_loader = DataLoader(TensorDataset(train_x, torch.zeros(len(train_x))),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, torch.zeros(len(val_x))),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SECsiNetRes(
        h=32, w=256, cr=args.cr, se_reduction=8,
        res_encoder_conv=args.res_encoder_conv,
        res_encoder_se=args.res_encoder_se,
        res_refine_se=args.res_refine_se
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    tag = f"cr{args.cr}_rec{int(args.res_encoder_conv)}_res{int(args.res_encoder_se)}_rrs{int(args.res_refine_se)}"
    ckpt_path = f"checkpoints/se_csinet_res_{tag}.pt"
    os.makedirs("checkpoints", exist_ok=True)

    best_nmse = 1e9

    # 可选：先跑统计
    if args.collect_stats:
        hook = ActivationStatsHook(save_dir=os.path.join(args.stats_dir, tag))
        hook.register(model)
        model.eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(train_loader):
                if i >= args.stats_batches:
                    break
                x = x.to(device)
                _ = model(x)
        csv_path = hook.save_csv("activation_stats_pretrain.csv")
        hook.remove()
        print(f"[Stats] 已保存: {csv_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        nmse_sum, cnt = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                pred = model(x)
                nmse_sum += nmse(pred, x).item() * x.size(0)
                cnt += x.size(0)
        val_nmse = nmse_sum / cnt
        print(f"Epoch {epoch:03d} | Loss {train_loss:.6f} | Val NMSE {val_nmse:.6f}")

        if val_nmse < best_nmse:
            best_nmse = val_nmse
            torch.save(model.state_dict(), ckpt_path)

    print(f"[Done] best nmse={best_nmse:.6f}, ckpt={ckpt_path}")

if __name__ == "__main__":
    main()
