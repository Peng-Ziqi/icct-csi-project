import os, argparse, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.se_jscc_learned_gate import SEJSCCLearnedGate
from utils.metrics import nmse
from utils.logger import setup_log_redirect

def seed_all(seed=2026):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor): x = obj
    elif isinstance(obj, dict): x = obj.get("x", obj.get("data", obj.get("H")))
    else: x = obj[0]
    if x.ndim == 5 and x.shape[2] == 1: x = x.squeeze(2)
    return x.float()

def snr_to_gate_class(snr_tensor, low_th=8.0, high_th=15.0):
    """
    class 0: full(1.0)    for snr < low_th
    class 1: half(0.5)    for low_th <= snr < high_th
    class 2: quarter(0.25)for snr >= high_th
    """
    cls = torch.zeros_like(snr_tensor, dtype=torch.long)
    cls = torch.where((snr_tensor >= low_th) & (snr_tensor < high_th), torch.ones_like(cls), cls)
    cls = torch.where(snr_tensor >= high_th, torch.full_like(cls, 2), cls)
    return cls.view(-1)

def class_to_target_ratio(cls):
    # 0->1.0, 1->0.5, 2->0.25
    table = torch.tensor([1.0, 0.5, 0.25], device=cls.device)
    return table[cls].view(-1, 1)

@torch.no_grad()
def evaluate(model, loader, device, snr_list, tau=0.2, low_th=8.0, high_th=15.0):
    model.eval()
    rows = []
    for snr in snr_list:
        s_nmse, s_k, n = 0.0, 0.0, 0
        gate_cnt = torch.zeros(3, device=device)
        for x, _ in loader:
            x = x.to(device)
            y, logits, gp, ratio, k = model(x, snr_db=snr, tau=tau, hard=True)
            s_nmse += nmse(y, x).item() * x.size(0)
            s_k += k.float().sum().item()
            n += x.size(0)

            cls = torch.argmax(gp, dim=1)  # [B]
            for i in range(3):
                gate_cnt[i] += (cls == i).float().sum()

        nmsev = s_nmse / n
        avgk = s_k / n
        over = avgk / model.comp_dim
        p = (gate_cnt / gate_cnt.sum()).detach().cpu().numpy().tolist()  # full, half, quarter
        rows.append((snr, nmsev, avgk, over, p[0], p[1], p[2]))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")

    # 分档阈值
    ap.add_argument("--low-th", type=float, default=8.0)
    ap.add_argument("--high-th", type=float, default=15.0)

    # loss系数
    ap.add_argument("--lambda-over", type=float, default=0.3)
    ap.add_argument("--lambda-gate", type=float, default=0.5)

    # Gumbel温度
    ap.add_argument("--tau-start", type=float, default=1.0)
    ap.add_argument("--tau-end", type=float, default=0.15)

    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/train_learned_gate_jscc_v2_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = [float(v) for v in args.snr_list.split(",")]

    tr = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    va = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))
    trl = DataLoader(TensorDataset(tr, torch.zeros(len(tr))), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TensorDataset(va, torch.zeros(len(va))), batch_size=args.batch_size, shuffle=False)

    model = SEJSCCLearnedGate(cr=args.cr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ckpt = f"checkpoints/learned_gate_jscc_v2_cr{args.cr}.pt"
    csvf = f"logs/train_learned_gate_jscc_v2_cr{args.cr}.csv"
    with open(csvf, "w") as f:
        f.write("epoch,train_loss,val_avg_nmse,val_avg_over,val_full,val_half,val_quarter\n")

    best = 1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        tau = args.tau_start + (args.tau_end - args.tau_start) * (ep - 1) / max(1, args.epochs - 1)

        ls, n = 0.0, 0
        for x, _ in trl:
            x = x.to(device)
            b = x.size(0)

            # 采样批次SNR（按样本独立采样，增强分档学习）
            snr_np = np.random.choice(snr_list, size=(b, 1))
            snr_t = torch.tensor(snr_np, dtype=torch.float32, device=device)

            y, logits, gp, ratio, k = model(x, snr_db=snr_t, tau=tau, hard=False)

            # 1) 重建
            loss_rec = mse(y, x)

            # 2) 门控分类监督
            cls = snr_to_gate_class(snr_t, low_th=args.low_th, high_th=args.high_th)  # [B]
            loss_gate = F.cross_entropy(logits, cls)

            # 3) 分组开销约束（按样本目标ratio）
            target_ratio = class_to_target_ratio(cls)  # [B,1]
            loss_over = ((ratio - target_ratio) ** 2).mean()

            loss = loss_rec + args.lambda_gate * loss_gate + args.lambda_over * loss_over

            opt.zero_grad()
            loss.backward()
            opt.step()

            ls += loss.item() * b
            n += b

        train_loss = ls / n

        # validation
        rows = evaluate(model, val, device, snr_list, tau=max(0.12, tau), low_th=args.low_th, high_th=args.high_th)
        avg_nmse = sum([r[1] for r in rows]) / len(rows)
        avg_over = sum([r[3] for r in rows]) / len(rows)
        avg_full = sum([r[4] for r in rows]) / len(rows)
        avg_half = sum([r[5] for r in rows]) / len(rows)
        avg_quarter = sum([r[6] for r in rows]) / len(rows)

        print(f"Epoch {ep:03d} | Loss {train_loss:.6f} | Val NMSE {avg_nmse:.6f} | Over {avg_over:.4f} "
              f"| GateDist F/H/Q={avg_full:.3f}/{avg_half:.3f}/{avg_quarter:.3f} | tau={tau:.3f}")

        with open(csvf, "a") as f:
            f.write(f"{ep},{train_loss:.8f},{avg_nmse:.8f},{avg_over:.6f},{avg_full:.6f},{avg_half:.6f},{avg_quarter:.6f}\n")

        if avg_nmse < best:
            best = avg_nmse
            torch.save(model.state_dict(), ckpt)

    print(f"[Done] best={best:.6f}, ckpt={ckpt}, csv={csvf}")

if __name__ == "__main__":
    main()
