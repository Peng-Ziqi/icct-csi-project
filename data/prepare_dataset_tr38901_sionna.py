import os
import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

import tensorflow as tf


def import_sionna_modules():
    """
    Target Sionna 1.2.2 with sionna.phy.* module paths.
    """
    try:
        from sionna.phy.channel.tr38901 import UMa, UMi, PanelArray
        from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel
        return {
            "UMa": UMa,
            "UMi": UMi,
            "PanelArray": PanelArray,
            "subcarrier_frequencies": subcarrier_frequencies,
            "cir_to_ofdm_channel": cir_to_ofdm_channel,
            "layout": "sionna.phy",
        }
    except Exception as e:
        raise ImportError(
            "Cannot import required Sionna TR 38.901 modules. "
            "Please check Sionna installation.\n"
            f"Error: {repr(e)}"
        )


SIONNA = import_sionna_modules()


SEED = 42

OUT_DIR = "data"
PROC_DIR = os.path.join(OUT_DIR, "processed")

CARRIER_FREQ = 3.5e9
SUBCARRIER_SPACING = 30e3
N_SUBCARRIERS = 256
BANDWIDTH = SUBCARRIER_SPACING * N_SUBCARRIERS

N_TX = 1
N_RX = 32

DEFAULT_SEQ_LEN = 8
DEFAULT_DT = 0.625e-3
DEFAULT_SPEEDS_KMH = [3.0, 30.0, 60.0]

SCENARIOS = ["UMa", "UMi"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


def kmh_to_mps(v_kmh):
    return float(v_kmh) / 3.6


def make_panel_array(num_ant, carrier_frequency):
    """
    Create a single-polarized omni panel array.

    Latest setting:
        TX antennas = 1  -> UT side for uplink
        RX antennas = 32 -> BS side for uplink

    We use uplink so the received CSI has 32 antenna channels.
    """
    PanelArray = SIONNA["PanelArray"]

    try:
        return PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
        )
    except TypeError:
        return PanelArray(
            num_rows_per_panel=1,
            num_cols_per_panel=num_ant,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=carrier_frequency,
            dtype=tf.complex64,
        )


def make_channel_model(
    scenario,
    carrier_frequency,
    bs_array,
    ut_array,
    direction="uplink",
    enable_pathloss=False,
    enable_shadow_fading=False,
):
    """
    Create a Sionna TR 38.901 UMa/UMi model.

    Latest antenna convention:
        Transmitter: UT, 1 antenna
        Receiver:    BS, 32 antennas

    Therefore, direction="uplink" is used.
    """
    if scenario == "UMa":
        Model = SIONNA["UMa"]
    elif scenario == "UMi":
        Model = SIONNA["UMi"]
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    kwargs = dict(
        carrier_frequency=carrier_frequency,
        ut_array=ut_array,
        bs_array=bs_array,
        direction=direction,
        enable_pathloss=enable_pathloss,
        enable_shadow_fading=enable_shadow_fading,
    )

    try:
        return Model(o2i_model="low", **kwargs)
    except TypeError:
        return Model(**kwargs)


def sample_uniform_annulus(batch_size, r_min, r_max):
    u = tf.random.uniform([batch_size], 0.0, 1.0, dtype=tf.float32)
    r = tf.sqrt((r_max ** 2 - r_min ** 2) * u + r_min ** 2)
    phi = tf.random.uniform([batch_size], -np.pi, np.pi, dtype=tf.float32)
    x = r * tf.cos(phi)
    y = r * tf.sin(phi)
    return x, y


def generate_topology(batch_size, scenario, speed_kmh):
    """
    Generate topology for Sionna TR 38.901 UMa/UMi.

    Shapes:
        ut_loc:          [B, 1, 3]
        bs_loc:          [B, 1, 3]
        ut_orientations: [B, 1, 3]
        bs_orientations: [B, 1, 3]
        ut_velocities:   [B, 1, 3]
        in_state:        [B, 1]

    Users are outdoor by default.
    """
    if scenario == "UMa":
        bs_height = 25.0
        ut_height = 1.5
        r_min = 35.0
        r_max = 500.0
    elif scenario == "UMi":
        bs_height = 10.0
        ut_height = 1.5
        r_min = 10.0
        r_max = 200.0
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    bs_x = tf.zeros([batch_size], dtype=tf.float32)
    bs_y = tf.zeros([batch_size], dtype=tf.float32)
    bs_z = tf.ones([batch_size], dtype=tf.float32) * bs_height

    ut_x, ut_y = sample_uniform_annulus(batch_size, r_min, r_max)
    ut_z = tf.ones([batch_size], dtype=tf.float32) * ut_height

    bs_loc = tf.stack([bs_x, bs_y, bs_z], axis=-1)
    ut_loc = tf.stack([ut_x, ut_y, ut_z], axis=-1)

    bs_loc = tf.expand_dims(bs_loc, axis=1)
    ut_loc = tf.expand_dims(ut_loc, axis=1)

    bs_orientations = tf.zeros([batch_size, 1, 3], dtype=tf.float32)
    ut_orientations = tf.zeros([batch_size, 1, 3], dtype=tf.float32)

    speed_mps = kmh_to_mps(speed_kmh)
    vel_angle = tf.random.uniform([batch_size], -np.pi, np.pi, dtype=tf.float32)
    vx = speed_mps * tf.cos(vel_angle)
    vy = speed_mps * tf.sin(vel_angle)
    vz = tf.zeros_like(vx)

    ut_vel = tf.stack([vx, vy, vz], axis=-1)
    ut_velocities = tf.expand_dims(ut_vel, axis=1)

    in_state = tf.zeros([batch_size, 1], dtype=tf.bool)

    return ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state


def set_topology(channel_model, topology):
    ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology

    try:
        channel_model.set_topology(
            ut_loc=ut_loc,
            bs_loc=bs_loc,
            ut_orientations=ut_orientations,
            bs_orientations=bs_orientations,
            ut_velocities=ut_velocities,
            in_state=in_state,
        )
    except TypeError:
        channel_model.set_topology(
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
        )


def call_channel_model(channel_model, batch_size, seq_len, sampling_frequency):
    """
    Generate time-varying CIR from Sionna channel model.
    Supports common call signatures across Sionna versions.
    """
    try:
        return channel_model(
            num_time_samples=seq_len,
            sampling_frequency=sampling_frequency,
        )
    except TypeError:
        pass

    try:
        return channel_model(seq_len, sampling_frequency)
    except TypeError:
        pass

    try:
        return channel_model(
            batch_size=batch_size,
            num_time_samples=seq_len,
            sampling_frequency=sampling_frequency,
        )
    except TypeError:
        pass

    try:
        return channel_model(batch_size, seq_len, sampling_frequency)
    except TypeError as e:
        raise RuntimeError(
            "Cannot call Sionna channel model with known signatures. "
            f"Last error: {repr(e)}"
        )


def extract_h_freq_to_torch(h_freq, seq_len):
    """
    Convert Sionna OFDM channel output to torch tensor [B,T,2,32,256].

    With uplink:
        TX side: UT, 1 antenna
        RX side: BS, 32 antennas

    Sionna OFDM channel is usually:
        [B, num_rx, num_rx_ant, num_tx, num_tx_ant, T, F]

    We robustly locate:
        T dimension = seq_len
        F dimension = 256
        RX antenna dimension = 32

    Output:
        [B, T, 2, 32, 256]
    """
    h_np = h_freq.numpy()
    h = torch.from_numpy(h_np)

    if not torch.is_complex(h):
        h = h.to(torch.complex64)

    shape = list(h.shape)

    if len(shape) < 4:
        raise RuntimeError(f"Unexpected h_freq shape: {shape}")

    t_candidates = [i for i, s in enumerate(shape) if s == seq_len and i != 0]
    if len(t_candidates) == 0:
        raise RuntimeError(f"Cannot find time dimension T={seq_len} in h_freq shape: {shape}")
    t_dim = t_candidates[-1]

    sc_candidates = [i for i, s in enumerate(shape) if s == N_SUBCARRIERS]
    if len(sc_candidates) == 0:
        raise RuntimeError(f"Cannot find subcarrier dimension {N_SUBCARRIERS} in h_freq shape: {shape}")
    sc_dim = sc_candidates[-1]

    rx_candidates = [
        i for i, s in enumerate(shape)
        if s == N_RX and i not in [0, t_dim, sc_dim]
    ]
    if len(rx_candidates) == 0:
        raise RuntimeError(f"Cannot find RX antenna dimension {N_RX} in h_freq shape: {shape}")
    rx_dim = rx_candidates[-1]

    front = [0, t_dim, rx_dim, sc_dim]
    rest = [i for i in range(len(shape)) if i not in front]
    h = h.permute(*(front + rest))

    squeeze_dims = list(range(4, h.ndim))
    for d in reversed(squeeze_dims):
        if h.shape[d] == 1:
            h = h.squeeze(d)

    if h.ndim != 4:
        raise RuntimeError(
            f"Unexpected shape after permutation/squeeze: {tuple(h.shape)} "
            f"from original shape {shape}"
        )

    x = torch.stack([h.real, h.imag], dim=2)
    return x.float()


def generate_batch(channel_model, scenario, speed_kmh, batch_size, seq_len, dt):
    """
    Generate one batch of time-varying frequency-domain CSI sequences.

    Output:
        x: [B,T,2,32,256]
    """
    topology = generate_topology(
        batch_size=batch_size,
        scenario=scenario,
        speed_kmh=speed_kmh,
    )
    set_topology(channel_model, topology)

    sampling_frequency = 1.0 / float(dt)

    a, tau = call_channel_model(
        channel_model=channel_model,
        batch_size=batch_size,
        seq_len=seq_len,
        sampling_frequency=sampling_frequency,
    )

    try:
        freqs = SIONNA["subcarrier_frequencies"](
            N_SUBCARRIERS,
            subcarrier_spacing=SUBCARRIER_SPACING,
        )
    except TypeError:
        freqs = SIONNA["subcarrier_frequencies"](
            N_SUBCARRIERS,
            SUBCARRIER_SPACING,
        )

    try:
        h_freq = SIONNA["cir_to_ofdm_channel"](
            freqs,
            a,
            tau,
            normalize=True,
        )
    except TypeError:
        h_freq = SIONNA["cir_to_ofdm_channel"](
            frequencies=freqs,
            a=a,
            tau=tau,
            normalize=True,
        )

    x = extract_h_freq_to_torch(h_freq, seq_len=seq_len)
    return x


def split_dataset(x, speed_kmh, scenario_id, seed):
    n = x.shape[0]

    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(n, generator=generator)

    x = x[idx]
    speed_kmh = speed_kmh[idx]
    scenario_id = scenario_id[idx]

    n_train = int(0.7 * n)
    n_val = int(0.2 * n)

    train = {
        "x": x[:n_train],
        "speed_kmh": speed_kmh[:n_train],
        "scenario_id": scenario_id[:n_train],
    }
    val = {
        "x": x[n_train:n_train + n_val],
        "speed_kmh": speed_kmh[n_train:n_train + n_val],
        "scenario_id": scenario_id[n_train:n_train + n_val],
    }
    test = {
        "x": x[n_train + n_val:],
        "speed_kmh": speed_kmh[n_train + n_val:],
        "scenario_id": scenario_id[n_train + n_val:],
    }

    return train, val, test


def normalize_by_train(train_obj, val_obj, test_obj):
    max_abs = torch.max(torch.abs(train_obj["x"]))
    if max_abs == 0:
        raise RuntimeError("Train max_abs is zero. Generated CSI is invalid.")

    train_obj["x"] = torch.clamp(train_obj["x"] / max_abs, -1.0, 1.0)
    val_obj["x"] = torch.clamp(val_obj["x"] / max_abs, -1.0, 1.0)
    test_obj["x"] = torch.clamp(test_obj["x"] / max_abs, -1.0, 1.0)

    return train_obj, val_obj, test_obj, max_abs


def print_checks(obj, name):
    x = obj["x"]
    print(f"{name} x:", tuple(x.shape))
    print(f"{name} speeds:", torch.unique(obj["speed_kmh"]))
    print(f"{name} scenarios:", torch.unique(obj["scenario_id"]))


def temporal_diff_by_speed(obj):
    x = obj["x"]
    speeds = obj["speed_kmh"]

    out = {}
    for speed in torch.unique(speeds):
        mask = speeds == speed
        xs = x[mask]
        if xs.shape[0] == 0:
            continue

        diffs = []
        for i in range(xs.shape[1] - 1):
            diff = ((xs[:, i + 1] - xs[:, i]) ** 2).mean().item()
            diffs.append(diff)

        out[float(speed.item())] = diffs

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-scenario-speed",
        type=int,
        default=3333,
        help="Samples for each scenario-speed pair.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--speeds-kmh", type=str, default="3,30,60")
    parser.add_argument("--scenarios", type=str, default="UMa,UMi")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--disable-pathloss",
        action="store_true",
        help="Disable pathloss. Recommended for normalized CSI feedback datasets.",
    )
    parser.add_argument(
        "--disable-shadow-fading",
        action="store_true",
        help="Disable shadow fading. Recommended for normalized CSI feedback datasets.",
    )
    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(PROC_DIR, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    speeds = [float(v) for v in args.speeds_kmh.split(",")]
    scenarios = [s.strip() for s in args.scenarios.split(",")]

    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unsupported scenario: {scenario}. Supported: {SCENARIOS}")

    expected_output_shape = f"[N,{args.seq_len},2,{N_RX},{N_SUBCARRIERS}]"

    print("[Sionna]")
    print("layout:", SIONNA["layout"])
    print("TensorFlow version:", tf.__version__)
    print("TF built with CUDA:", tf.test.is_built_with_cuda())
    print("TF GPUs:", tf.config.list_physical_devices("GPU"))

    print("\n[Config]")
    print("samples_per_scenario_speed:", args.samples_per_scenario_speed)
    print("batch_size:", args.batch_size)
    print("carrier_freq:", CARRIER_FREQ)
    print("subcarrier_spacing:", SUBCARRIER_SPACING)
    print("n_subcarriers:", N_SUBCARRIERS)
    print("effective_bandwidth:", BANDWIDTH)
    print("n_tx:", N_TX)
    print("n_rx:", N_RX)
    print("seq_len:", args.seq_len)
    print("dt:", args.dt)
    print("speeds_kmh:", speeds)
    print("scenarios:", scenarios)
    print("disable_pathloss:", args.disable_pathloss)
    print("disable_shadow_fading:", args.disable_shadow_fading)
    print("direction:", "uplink")
    print("output shape per sample:", expected_output_shape.replace("N,", ""))

    bs_array = make_panel_array(N_RX, CARRIER_FREQ)
    ut_array = make_panel_array(N_TX, CARRIER_FREQ)

    all_x = []
    all_speed = []
    all_scenario_id = []

    for scenario in scenarios:
        channel_model = make_channel_model(
            scenario=scenario,
            carrier_frequency=CARRIER_FREQ,
            bs_array=bs_array,
            ut_array=ut_array,
            direction="uplink",
            enable_pathloss=(not args.disable_pathloss),
            enable_shadow_fading=(not args.disable_shadow_fading),
        )

        scenario_id = SCENARIOS.index(scenario)

        for speed in speeds:
            n_left = args.samples_per_scenario_speed
            pbar = tqdm(total=n_left, desc=f"Generating {scenario}, {speed:g} km/h")

            while n_left > 0:
                batch_size = min(args.batch_size, n_left)

                x_b = generate_batch(
                    channel_model=channel_model,
                    scenario=scenario,
                    speed_kmh=speed,
                    batch_size=batch_size,
                    seq_len=args.seq_len,
                    dt=args.dt,
                )

                expected_shape = (batch_size, args.seq_len, 2, N_RX, N_SUBCARRIERS)
                if tuple(x_b.shape) != expected_shape:
                    raise RuntimeError(
                        f"Unexpected generated shape: {tuple(x_b.shape)}, "
                        f"expected {expected_shape}"
                    )

                all_x.append(x_b.cpu())
                all_speed.append(torch.full((batch_size,), float(speed), dtype=torch.float32))
                all_scenario_id.append(torch.full((batch_size,), int(scenario_id), dtype=torch.long))

                n_left -= batch_size
                pbar.update(batch_size)

            pbar.close()

    x = torch.cat(all_x, dim=0)
    speed_kmh = torch.cat(all_speed, dim=0)
    scenario_id = torch.cat(all_scenario_id, dim=0)

    train_obj, val_obj, test_obj = split_dataset(
        x=x,
        speed_kmh=speed_kmh,
        scenario_id=scenario_id,
        seed=args.seed,
    )

    train_obj, val_obj, test_obj, max_abs = normalize_by_train(
        train_obj=train_obj,
        val_obj=val_obj,
        test_obj=test_obj,
    )

    train_path = os.path.join(PROC_DIR, "train_tr38901_seq.pt")
    val_path = os.path.join(PROC_DIR, "val_tr38901_seq.pt")
    test_path = os.path.join(PROC_DIR, "test_tr38901_seq.pt")
    meta_path = os.path.join(PROC_DIR, "meta_tr38901_seq.json")

    torch.save(train_obj, train_path)
    torch.save(val_obj, val_path)
    torch.save(test_obj, test_path)

    meta = {
        "dataset_type": "standard_tr38901_time_varying_csi_sequence",
        "tool": "Sionna",
        "sionna_layout": SIONNA["layout"],
        "tensorflow_version": tf.__version__,
        "description": (
            "Time-varying CSI sequences generated by Sionna TR 38.901 "
            "UMa/UMi channel models with uplink direction, UT mobility, "
            "1 transmit antenna, and 32 receive antennas."
        ),
        "n_total": int(x.shape[0]),
        "n_train": int(train_obj["x"].shape[0]),
        "n_val": int(val_obj["x"].shape[0]),
        "n_test": int(test_obj["x"].shape[0]),
        "samples_per_scenario_speed": int(args.samples_per_scenario_speed),
        "scenarios": scenarios,
        "scenario_id_mapping": {str(i): s for i, s in enumerate(SCENARIOS)},
        "speeds_kmh": speeds,
        "seq_len": int(args.seq_len),
        "dt": float(args.dt),
        "carrier_freq": float(CARRIER_FREQ),
        "subcarrier_spacing": float(SUBCARRIER_SPACING),
        "n_subcarriers": int(N_SUBCARRIERS),
        "effective_bandwidth": float(BANDWIDTH),
        "n_tx": int(N_TX),
        "n_rx": int(N_RX),
        "direction": "uplink",
        "bs_height_UMa": 25.0,
        "bs_height_UMi": 10.0,
        "ut_height": 1.5,
        "topology": "manual outdoor single-BS single-UT topology for Sionna TR 38.901 UMa/UMi",
        "pathloss_enabled": bool(not args.disable_pathloss),
        "shadow_fading_enabled": bool(not args.disable_shadow_fading),
        "normalization": "train-set max_abs normalization",
        "max_abs": float(max_abs.item()),
        "split": "train/val/test = 7/2/1",
        "seed": int(args.seed),
        "output_shape": expected_output_shape,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[Check]")
    print_checks(train_obj, "train")
    print_checks(val_obj, "val")
    print_checks(test_obj, "test")

    diff_by_speed = temporal_diff_by_speed(test_obj)
    print("\n[Temporal Diff By Speed on Test]")
    for speed in sorted(diff_by_speed.keys()):
        print(f"{speed:g} km/h:", diff_by_speed[speed])

    print("\n[Done]")
    print("Saved:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    print(" -", meta_path)


if __name__ == "__main__":
    main()
