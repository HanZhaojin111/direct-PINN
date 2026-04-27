#!/usr/bin/env python3
import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MESH_TOLERANCE = 1e-6
MIN_RANGE_THRESHOLD = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 2D Navier-Stokes PINN on VTU time series and predict future steps."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing .vtu files.")
    parser.add_argument("--train-steps", type=int, default=601, help="Number of time steps to train on.")
    parser.add_argument("--predict-steps", type=int, default=200, help="Number of future steps to predict.")
    parser.add_argument("--velocity-key", default=None, help="VTU point_data key for velocity.")
    parser.add_argument("--pressure-key", default=None, help="VTU point_data key for pressure.")
    parser.add_argument(
        "--max-points-per-time",
        type=int,
        default=5000,
        help="Max sampled points per time step for supervised data.",
    )
    parser.add_argument("--nu", type=float, default=0.01, help="Kinematic viscosity.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Supervised batch size.")
    parser.add_argument(
        "--collocation-points",
        type=int,
        default=20000,
        help="Number of collocation points sampled per epoch.",
    )
    parser.add_argument("--hidden-layers", type=int, default=6, help="Number of hidden layers.")
    parser.add_argument("--hidden-width", type=int, default=128, help="Width of hidden layers.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--lambda-data", type=float, default=1.0, help="Weight for data loss.")
    parser.add_argument("--lambda-phys", type=float, default=1.0, help="Weight for PDE residual loss.")
    parser.add_argument("--lambda-div", type=float, default=1.0, help="Weight for divergence loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda.")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N epochs.")
    parser.add_argument("--output-dir", default="predictions", help="Output directory for predicted VTU files.")
    parser.add_argument("--output-velocity-key", default="velocity", help="Velocity key for outputs.")
    parser.add_argument("--output-pressure-key", default="pressure", help="Pressure key for outputs.")
    parser.add_argument("--save-model", default=None, help="Path to save trained model state_dict.")
    parser.add_argument("--pred-batch-size", type=int, default=16384, help="Batch size for prediction.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_time_index(filename: str) -> Optional[float]:
    match = re.search(r"_(-?\d+(?:\.\d+)?)\.vtu$", os.path.basename(filename))
    if match:
        return float(match.group(1))
    return None


def list_vtu_files(data_dir: str) -> Tuple[List[str], np.ndarray]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.vtu")))
    if not files:
        raise FileNotFoundError(f"No .vtu files found in {data_dir}")
    indexed = [(extract_time_index(path), path) for path in files]
    if all(idx is not None for idx, _ in indexed):
        indexed.sort(key=lambda item: item[0])
        times = np.array([idx for idx, _ in indexed], dtype=np.float32)
        files = [path for _, path in indexed]
    else:
        files.sort()
        times = np.arange(len(files), dtype=np.float32)
    return files, times


def resolve_point_data_key(
    point_data: Dict[str, np.ndarray],
    requested: Optional[str],
    candidates: List[str],
    label: str,
) -> str:
    lower_map = {key.lower(): key for key in point_data.keys()}
    if requested:
        if requested in point_data:
            return requested
        requested_lower = requested.lower()
        if requested_lower in lower_map:
            return lower_map[requested_lower]
        raise KeyError(f"{label} key '{requested}' not found in point_data.")
    for candidate in candidates:
        if candidate in point_data:
            return candidate
        candidate_lower = candidate.lower()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]
    raise KeyError(f"Unable to infer {label} key from point_data: {list(point_data.keys())}")


def read_vtu(
    file_path: str,
    velocity_key: Optional[str],
    pressure_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, meshio.Mesh]:
    mesh = meshio.read(file_path)
    points = mesh.points[:, :2].astype(np.float32)
    vel_key = resolve_point_data_key(
        mesh.point_data, velocity_key, ["velocity", "Velocity", "U", "u"], "velocity"
    )
    pre_key = resolve_point_data_key(
        mesh.point_data, pressure_key, ["pressure", "Pressure", "p", "P"], "pressure"
    )
    velocity = np.asarray(mesh.point_data[vel_key], dtype=np.float32)
    if velocity.ndim != 2 or velocity.shape[1] < 2:
        raise ValueError(f"Velocity data must be 2D with at least 2 components, got {velocity.shape}")
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    pressure = np.asarray(mesh.point_data[pre_key], dtype=np.float32).reshape(-1, 1)
    return points, u, v, pressure, mesh


def sample_supervised_data(
    points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    total_points = points.shape[0]
    if max_points > 0 and total_points > max_points:
        idx = rng.choice(total_points, size=max_points, replace=False)
        points = points[idx]
        u = u[idx]
        v = v[idx]
        p = p[idx]
    outputs = np.hstack([u, v, p]).astype(np.float32)
    return points.astype(np.float32), outputs


def build_training_data(
    files: List[str],
    times: np.ndarray,
    train_steps: int,
    velocity_key: Optional[str],
    pressure_key: Optional[str],
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, meshio.Mesh, np.ndarray]:
    if train_steps > len(files):
        raise ValueError(f"train_steps={train_steps} exceeds number of files {len(files)}")
    xyt_list = []
    uvp_list = []
    base_mesh: Optional[meshio.Mesh] = None
    base_points: Optional[np.ndarray] = None
    for idx in range(train_steps):
        points, u, v, p, mesh = read_vtu(files[idx], velocity_key, pressure_key)
        if base_mesh is None:
            base_mesh = mesh
            base_points = points
        elif base_points is not None:
            if base_points.shape != points.shape:
                raise ValueError(
                    f"Mesh shape mismatch: expected {base_points.shape}, got {points.shape}"
                )
            if not np.allclose(base_points, points, atol=MESH_TOLERANCE):
                raise ValueError("Inconsistent mesh points across time steps.")
        points_sampled, outputs = sample_supervised_data(points, u, v, p, max_points, rng)
        t_col = np.full((points_sampled.shape[0], 1), times[idx], dtype=np.float32)
        xyt_list.append(np.hstack([points_sampled, t_col]))
        uvp_list.append(outputs)
    if base_mesh is None:
        raise RuntimeError("Failed to load base mesh from VTU files.")
    xyt = np.vstack(xyt_list).astype(np.float32)
    uvp = np.vstack(uvp_list).astype(np.float32)
    return xyt, uvp, base_mesh, times[:train_steps]


class Scaler:
    def __init__(self, data_min: np.ndarray, data_max: np.ndarray) -> None:
        self.data_min = data_min.astype(np.float32)
        self.data_max = data_max.astype(np.float32)
        self.data_range = np.maximum(self.data_max - self.data_min, MIN_RANGE_THRESHOLD)
        self.scale = 2.0 / self.data_range

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.data_min) * self.scale - 1.0

    def transform_torch(self, values: torch.Tensor) -> torch.Tensor:
        return (values - torch.from_numpy(self.data_min).to(values.device)) * torch.from_numpy(
            self.scale
        ).to(values.device) - 1.0

    def scale_factors(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.scale).to(device)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def pde_residuals(
    model: nn.Module,
    xyt_norm: torch.Tensor,
    scale: torch.Tensor,
    nu: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyt_norm.requires_grad_(True)
    pred = model(xyt_norm)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    grads_u = torch.autograd.grad(u, xyt_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, xyt_norm, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    grads_p = torch.autograd.grad(p, xyt_norm, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_x_hat, u_y_hat, u_t_hat = grads_u[:, 0:1], grads_u[:, 1:2], grads_u[:, 2:3]
    v_x_hat, v_y_hat, v_t_hat = grads_v[:, 0:1], grads_v[:, 1:2], grads_v[:, 2:3]
    p_x_hat, p_y_hat = grads_p[:, 0:1], grads_p[:, 1:2]

    u_xx_hat = torch.autograd.grad(
        u_x_hat, xyt_norm, grad_outputs=torch.ones_like(u_x_hat), create_graph=True
    )[0][:, 0:1]
    u_yy_hat = torch.autograd.grad(
        u_y_hat, xyt_norm, grad_outputs=torch.ones_like(u_y_hat), create_graph=True
    )[0][:, 1:2]
    v_xx_hat = torch.autograd.grad(
        v_x_hat, xyt_norm, grad_outputs=torch.ones_like(v_x_hat), create_graph=True
    )[0][:, 0:1]
    v_yy_hat = torch.autograd.grad(
        v_y_hat, xyt_norm, grad_outputs=torch.ones_like(v_y_hat), create_graph=True
    )[0][:, 1:2]

    sx, sy, st = scale[0], scale[1], scale[2]
    u_x = u_x_hat * sx
    u_y = u_y_hat * sy
    u_t = u_t_hat * st
    v_x = v_x_hat * sx
    v_y = v_y_hat * sy
    v_t = v_t_hat * st
    p_x = p_x_hat * sx
    p_y = p_y_hat * sy
    u_xx = u_xx_hat * (sx**2)
    u_yy = u_yy_hat * (sy**2)
    v_xx = v_xx_hat * (sx**2)
    v_yy = v_yy_hat * (sy**2)

    res_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    res_div = u_x + v_y
    return res_u, res_v, res_div


def sample_collocation(
    rng: np.random.Generator,
    n_points: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    mins, maxs = bounds
    points = rng.uniform(mins, maxs, size=(n_points, 3)).astype(np.float32)
    return torch.tensor(points, device=device)


def train_pinn(
    model: nn.Module,
    data_loader: DataLoader,
    scaler: Scaler,
    nu: float,
    epochs: int,
    collocation_points: int,
    lambda_data: float,
    lambda_phys: float,
    lambda_div: float,
    bounds: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    rng: np.random.Generator,
    log_every: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scale = scaler.scale_factors(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xyt_batch, uvp_batch in data_loader:
            xyt_batch = xyt_batch.to(device)
            uvp_batch = uvp_batch.to(device)
            optimizer.zero_grad()
            pred = model(scaler.transform_torch(xyt_batch))
            data_loss = torch.mean((pred - uvp_batch) ** 2)

            xyt_f = sample_collocation(rng, collocation_points, bounds, device)
            xyt_f_norm = scaler.transform_torch(xyt_f)
            res_u, res_v, res_div = pde_residuals(model, xyt_f_norm, scale, nu)
            phys_loss = torch.mean(res_u**2) + torch.mean(res_v**2)
            div_loss = torch.mean(res_div**2)

            loss = lambda_data * data_loss + lambda_phys * phys_loss + lambda_div * div_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if log_every > 0 and epoch % log_every == 0:
            avg_loss = epoch_loss / max(len(data_loader), 1)
            print(
                f"Epoch {epoch}/{epochs} | loss={avg_loss:.6e} "
                f"data={data_loss.item():.6e} phys={phys_loss.item():.6e} div={div_loss.item():.6e}"
            )


def predict_future(
    model: nn.Module,
    scaler: Scaler,
    base_mesh: meshio.Mesh,
    times: np.ndarray,
    predict_steps: int,
    output_dir: str,
    velocity_key: str,
    pressure_key: str,
    pred_batch_size: int,
    device: torch.device,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    points = base_mesh.points[:, :2].astype(np.float32)
    if len(times) < 2:
        dt = 1.0
    else:
        dt = float(np.median(np.diff(times)))
    start_t = float(times[-1])
    future_times = start_t + dt * np.arange(1, predict_steps + 1, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for step, t_value in enumerate(future_times, start=1):
            outputs = []
            for i in range(0, points.shape[0], pred_batch_size):
                chunk = points[i : i + pred_batch_size]
                t_col = np.full((chunk.shape[0], 1), t_value, dtype=np.float32)
                xyt = np.hstack([chunk, t_col]).astype(np.float32)
                xyt_t = torch.tensor(xyt, device=device)
                pred = model(scaler.transform_torch(xyt_t)).cpu().numpy()
                outputs.append(pred)
            pred_all = np.vstack(outputs)
            velocity = np.zeros((pred_all.shape[0], 3), dtype=np.float32)
            velocity[:, 0:2] = pred_all[:, 0:2]
            pressure = pred_all[:, 2:3]

            mesh_out = meshio.Mesh(
                points=base_mesh.points,
                cells=base_mesh.cells,
                point_data={
                    velocity_key: velocity,
                    pressure_key: pressure,
                },
            )
            filename = os.path.join(output_dir, f"pred_{step:04d}.vtu")
            meshio.write(filename, mesh_out)
            print(f"Saved {filename}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)

    files, times = list_vtu_files(args.data_dir)
    xyt, uvp, base_mesh, train_times = build_training_data(
        files,
        times,
        args.train_steps,
        args.velocity_key,
        args.pressure_key,
        args.max_points_per_time,
        rng,
    )

    data_min = np.min(xyt, axis=0)
    data_max = np.max(xyt, axis=0)
    scaler = Scaler(data_min, data_max)
    bounds = (data_min, data_max)

    dataset = TensorDataset(torch.tensor(xyt), torch.tensor(uvp))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = MLP(3, 3, args.hidden_width, args.hidden_layers).to(device)
    train_pinn(
        model,
        data_loader,
        scaler,
        args.nu,
        args.epochs,
        args.collocation_points,
        args.lambda_data,
        args.lambda_phys,
        args.lambda_div,
        bounds,
        device,
        rng,
        args.log_every,
        args.learning_rate,
    )

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved model to {args.save_model}")

    predict_future(
        model,
        scaler,
        base_mesh,
        train_times,
        args.predict_steps,
        args.output_dir,
        args.output_velocity_key,
        args.output_pressure_key,
        args.pred_batch_size,
        device,
    )


if __name__ == "__main__":
    main()
