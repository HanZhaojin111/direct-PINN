import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import meshio
except ImportError as exc:
    raise SystemExit("meshio is required. Install with: pip install meshio") from exc


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_vtu_files(data_dir: Path, prefix: str) -> list[tuple[int, Path]]:
    prefix = Path(prefix).stem
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.vtu$")
    files = []
    for path in data_dir.glob(f"{prefix}_*.vtu"):
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return files


def find_point_data_key(point_data: dict, candidates: list[str]) -> str | None:
    lower_map = {key.lower(): key for key in point_data.keys()}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            return key
    return None


def load_vtu_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, meshio.Mesh]:
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)
    point_data = mesh.point_data or {}

    pressure_key = find_point_data_key(point_data, ["Pressure", "pressure", "p"])
    velocity_key = find_point_data_key(point_data, ["Velocity", "velocity", "u"])
    time_key = find_point_data_key(point_data, ["Time", "time"])

    if pressure_key is None or velocity_key is None:
        raise ValueError(f"Missing Pressure or Velocity in {path}")

    pressure = np.asarray(point_data[pressure_key]).reshape(-1)
    velocity = np.asarray(point_data[velocity_key])
    if velocity.ndim == 1:
        velocity = velocity.reshape(-1, 1)
    if velocity.shape[1] < 2:
        raise ValueError(f"Velocity needs at least 2 components in {path}")
    velocity = velocity[:, :2]

    time_value = None
    if time_key is not None:
        time_array = np.asarray(point_data[time_key]).reshape(-1)
        if time_array.size > 0:
            time_value = float(time_array[0])

    return points[:, :2], pressure, velocity, time_value, mesh


def infer_dt(times: list[float | None]) -> float | None:
    values = sorted({t for t in times if t is not None})
    if len(values) < 2:
        return None
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int, hidden_units: int) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gradient(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def compute_residuals(
    model: nn.Module,
    xyt: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    nu: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyt_norm = normalize(xyt, x_mean, x_std)
    xyt_norm.requires_grad_(True)
    pred = denormalize(model(xyt_norm), y_mean, y_std)

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    grads_u = gradient(u, xyt_norm)
    grads_v = gradient(v, xyt_norm)
    grads_p = gradient(p, xyt_norm)

    inv_std = 1.0 / x_std
    du_dx = grads_u[:, 0:1] * inv_std[:, 0:1]
    du_dy = grads_u[:, 1:2] * inv_std[:, 1:2]
    du_dt = grads_u[:, 2:3] * inv_std[:, 2:3]

    dv_dx = grads_v[:, 0:1] * inv_std[:, 0:1]
    dv_dy = grads_v[:, 1:2] * inv_std[:, 1:2]
    dv_dt = grads_v[:, 2:3] * inv_std[:, 2:3]

    dp_dx = grads_p[:, 0:1] * inv_std[:, 0:1]
    dp_dy = grads_p[:, 1:2] * inv_std[:, 1:2]

    d2u_dx2 = gradient(grads_u[:, 0:1], xyt_norm)[:, 0:1] * (inv_std[:, 0:1] ** 2)
    d2u_dy2 = gradient(grads_u[:, 1:2], xyt_norm)[:, 1:2] * (inv_std[:, 1:2] ** 2)
    d2v_dx2 = gradient(grads_v[:, 0:1], xyt_norm)[:, 0:1] * (inv_std[:, 0:1] ** 2)
    d2v_dy2 = gradient(grads_v[:, 1:2], xyt_norm)[:, 1:2] * (inv_std[:, 1:2] ** 2)

    f_u = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    f_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
    f_c = du_dx + dv_dy
    return f_u, f_v, f_c


def train_pinn(
    model: nn.Module,
    loader: DataLoader,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    nu: float,
    epochs: int,
    lr: float,
    lambda_phys: float,
    device: torch.device,
    log_every: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xyt_batch, uvp_batch in loader:
            xyt_batch = xyt_batch.to(device)
            uvp_batch = uvp_batch.to(device)

            optimizer.zero_grad()
            pred = denormalize(model(normalize(xyt_batch, x_mean, x_std)), y_mean, y_std)
            data_loss = mse(pred, uvp_batch)

            f_u, f_v, f_c = compute_residuals(model, xyt_batch, x_mean, x_std, y_mean, y_std, nu)
            phys_loss = (f_u.pow(2).mean() + f_v.pow(2).mean() + f_c.pow(2).mean())

            loss = data_loss + lambda_phys * phys_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if log_every > 0 and epoch % log_every == 0:
            avg_loss = total_loss / max(1, len(loader))
            print(f"Epoch {epoch:6d} | Loss {avg_loss:.6e}")


def predict_future(
    model: nn.Module,
    mesh_template: meshio.Mesh,
    times: list[float],
    output_dir: Path,
    output_prefix: str,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> None:
    points = np.asarray(mesh_template.points, dtype=np.float64)
    xy = points[:, :2]
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for step, time_value in enumerate(times, start=1):
            xyt = np.column_stack([xy, np.full((xy.shape[0], 1), time_value)])
            xyt_tensor = torch.from_numpy(xyt).float().to(device)
            pred = denormalize(model(normalize(xyt_tensor, x_mean, x_std)), y_mean, y_std)
            pred_np = pred.cpu().numpy()
            u = pred_np[:, 0]
            v = pred_np[:, 1]
            p = pred_np[:, 2]
            velocity = np.column_stack([u, v, np.zeros_like(u)])

            mesh = meshio.Mesh(
                points=mesh_template.points,
                cells=mesh_template.cells,
                point_data={"Pressure": p, "Velocity": velocity},
            )
            out_path = output_dir / f"{output_prefix}_{step}.vtu"
            meshio.write(out_path, mesh)


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN for 2D unsteady Navier-Stokes with VTU data")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory with VTU files")
    parser.add_argument("--prefix", type=str, default="circle-2d-drag", help="VTU file prefix")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive)")
    parser.add_argument("--end-index", type=int, default=None, help="End index (inclusive)")
    parser.add_argument("--predict-steps", type=int, default=200, help="Number of future steps to predict")
    parser.add_argument("--dt", type=float, default=None, help="Time step size if not in VTU")
    parser.add_argument("--nu", type=float, default=1e-3, help="Kinematic viscosity")
    parser.add_argument("--max-points", type=int, default=0, help="Max points per time step (0 = all)")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-phys", type=float, default=1.0, help="Physics loss weight")
    parser.add_argument("--hidden-layers", type=int, default=6, help="Hidden layers")
    parser.add_argument("--hidden-units", type=int, default=64, help="Hidden units")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N epochs (0=disable)")
    parser.add_argument("--output-dir", type=Path, default=Path("predicted_vtu"), help="Output directory")
    parser.add_argument("--output-prefix", type=str, default="predicted", help="Output VTU prefix")
    parser.add_argument("--model-out", type=Path, default=Path("pinn_ns.pt"), help="Model output path")
    parser.add_argument("--stats-out", type=Path, default=Path("pinn_ns_stats.npz"), help="Normalization stats output")
    args = parser.parse_args()

    set_seed(args.seed)

    files = discover_vtu_files(args.data_dir, args.prefix)
    if not files:
        raise SystemExit(f"No VTU files found in {args.data_dir} with prefix {args.prefix}")

    if args.start_index is not None:
        files = [item for item in files if item[0] >= args.start_index]
    if args.end_index is not None:
        files = [item for item in files if item[0] <= args.end_index]
    if not files:
        raise SystemExit("No VTU files left after applying index filters")

    rng = np.random.default_rng(args.seed)
    max_points = args.max_points if args.max_points > 0 else None

    snapshots = []
    time_values = []
    mesh_template = None
    for index, path in files:
        xy, pressure, velocity, time_value, mesh = load_vtu_snapshot(path)
        mesh_template = mesh_template or mesh
        snapshots.append((index, xy, pressure, velocity))
        time_values.append(time_value)

    inferred_dt = infer_dt(time_values)
    dt = args.dt if args.dt is not None else inferred_dt
    if dt is None:
        dt = 1.0
    if not math.isfinite(dt) or dt <= 0:
        raise SystemExit("Invalid dt detected or provided")

    xyt_list = []
    uvp_list = []
    for (index, xy, pressure, velocity), time_value in zip(snapshots, time_values):
        if time_value is None:
            time_value = index * dt
        if max_points is None or xy.shape[0] <= max_points:
            sample_idx = np.arange(xy.shape[0])
        else:
            sample_idx = rng.choice(xy.shape[0], size=max_points, replace=False)
        xy_sample = xy[sample_idx]
        u_sample = velocity[sample_idx, 0]
        v_sample = velocity[sample_idx, 1]
        p_sample = pressure[sample_idx]
        t_sample = np.full((xy_sample.shape[0], 1), time_value)
        xyt_list.append(np.column_stack([xy_sample, t_sample]))
        uvp_list.append(np.column_stack([u_sample, v_sample, p_sample]))

    xyt_all = np.vstack(xyt_list).astype(np.float32)
    uvp_all = np.vstack(uvp_list).astype(np.float32)

    x_mean = torch.from_numpy(xyt_all.mean(axis=0, keepdims=True))
    x_std = torch.from_numpy(xyt_all.std(axis=0, keepdims=True) + 1e-8)
    y_mean = torch.from_numpy(uvp_all.mean(axis=0, keepdims=True))
    y_std = torch.from_numpy(uvp_all.std(axis=0, keepdims=True) + 1e-8)

    dataset = TensorDataset(torch.from_numpy(xyt_all), torch.from_numpy(uvp_all))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=3, out_dim=3, hidden_layers=args.hidden_layers, hidden_units=args.hidden_units).to(device)
    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    train_pinn(
        model=model,
        loader=loader,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        nu=args.nu,
        epochs=args.epochs,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
        device=device,
        log_every=args.log_every,
    )

    torch.save(model.state_dict(), args.model_out)
    np.savez(
        args.stats_out,
        x_mean=x_mean.detach().cpu().numpy(),
        x_std=x_std.detach().cpu().numpy(),
        y_mean=y_mean.detach().cpu().numpy(),
        y_std=y_std.detach().cpu().numpy(),
        dt=dt,
    )

    valid_times = [t for t in time_values if t is not None]
    last_time = max(valid_times) if valid_times else files[-1][0] * dt
    future_times = [last_time + dt * step for step in range(1, args.predict_steps + 1)]

    if mesh_template is None:
        raise SystemExit("Failed to load mesh template")

    predict_future(
        model=model,
        mesh_template=mesh_template,
        times=future_times,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
    )


if __name__ == "__main__":
    main()
