import os.path
import sys
import os
sys.path.insert(0, '../deep_learning')
import math
import argparse
import random
import time
import numpy as np
import logging
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import zarr

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import wandb

def _is_cluster() -> bool:
    return (
        "SLURM_JOB_ID" in os.environ
        or "SLURM_CLUSTER_NAME" in os.environ
        or "SBATCH_JOB_ID" in os.environ
    )

def _apply_auto_paths(opt):
    local_nc = "C:/users/david/PythonProjekte/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/hrrr_mini/hrrr_mini_train.nc"
    local_stats = "C:/users/david/PythonProjekte/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/hrrr_mini/stats.json"

    cluster_nc = "/home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/modulus_datasets-hrrr_mini_v1/hrrr_mini/hrrr_mini_train.nc"
    cluster_stats = "/home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/modulus_datasets-hrrr_mini_v1/hrrr_mini/stats.json"

    use_cluster = _is_cluster()

    for phase in ["train", "test"]:
        if "datasets" not in opt or phase not in opt["datasets"]:
            continue

        ds = opt["datasets"][phase]

        if use_cluster:
            ds["dataroot_nc"] = cluster_nc
            ds["stats_path"] = cluster_stats
        else:
            ds["dataroot_nc"] = local_nc
            ds["stats_path"] = local_stats

    return opt

def _add_physicsnemo_root():
    here = os.path.abspath(os.path.dirname(__file__))
    p = here
    for _ in range(20):
        cand = os.path.join(p, "examples", "weather", "corrdiff")
        if os.path.isdir(cand):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
        p = os.path.dirname(p)
    return None

_add_physicsnemo_root()

def _add_corrdiff_root():
    here = os.path.abspath(os.path.dirname(__file__))
    p = here
    for _ in range(20):
        corrdiff_root = os.path.join(p, "examples", "weather", "corrdiff")
        if os.path.isdir(os.path.join(corrdiff_root, "datasets")):
            if corrdiff_root not in sys.path:
                sys.path.insert(0, corrdiff_root)
            return corrdiff_root
        p = os.path.dirname(p)
    return None

_add_corrdiff_root()

def _apply_data_path_override(opt):
    local_path = os.environ.get("CWA_DATA_PATH")
    if not local_path:
        return opt
    if "datasets" not in opt:
        return opt
    for phase in ["train", "test"]:
        if phase in opt["datasets"]:
            opt["datasets"][phase]["data_path"] = local_path
    return opt

def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(x).float()


def _sanitize_logs(logs):
    clean = {}
    for k, v in logs.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.detach().float().mean().item()
        else:
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
    return clean


def _build_cwb_cfg(dataset_opt, train_flag):
    # Avoid leakage for validation: force split-aware loading for test side.
    all_times = bool(dataset_opt.get("all_times", False))
    if not train_flag and all_times:
        all_times = False

    cfg = {
        "data_path": dataset_opt["data_path"],
        "normalization": dataset_opt.get("normalization", "v1"),
        "all_times": all_times,
        "train": train_flag,
        "in_channels": dataset_opt.get("in_channels", None),
        "out_channels": dataset_opt.get("out_channels", None),
        "img_shape_x": dataset_opt.get("img_shape_x", 448),
        "img_shape_y": dataset_opt.get("img_shape_y", 448),
        "add_grid": dataset_opt.get("add_grid", True),
        "ds_factor": dataset_opt.get("ds_factor", 1),
        "n_history": dataset_opt.get("n_history", 0),
        "min_path": dataset_opt.get("min_path", None),
        "max_path": dataset_opt.get("max_path", None),
        "global_means_path": dataset_opt.get("global_means_path", None),
        "global_stds_path": dataset_opt.get("global_stds_path", None),
    }
    return {k: v for k, v in cfg.items() if v is not None}


def _make_train_loader(train_set, dataset_opt, opt, seed):
    if opt['dist']:
        train_sampler = DistributedSampler(
            train_set,
            shuffle=dataset_opt['dataloader_shuffle'],
            drop_last=True,
            seed=seed,
        )
        train_batch_size = max(1, dataset_opt['dataloader_batch_size'] // opt['num_gpu'])
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
            drop_last=True,
            pin_memory=True,
            persistent_workers=(dataset_opt['dataloader_num_workers'] // opt['num_gpu']) > 0,
            prefetch_factor=2 if (dataset_opt['dataloader_num_workers'] // opt['num_gpu']) > 0 else None,
            sampler=train_sampler,
        )
    else:
        train_sampler = None
        train_batch_size = dataset_opt['dataloader_batch_size']
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=dataset_opt['dataloader_shuffle'],
            num_workers=dataset_opt['dataloader_num_workers'],
            drop_last=True,
            pin_memory=True,
            persistent_workers=dataset_opt['dataloader_num_workers'] > 0,
            prefetch_factor=2 if dataset_opt['dataloader_num_workers'] > 0 else None,
        )
    return train_loader, train_sampler, train_batch_size


def _make_test_loader(test_set, dataset_opt, test_batch_size):
    return DataLoader(
        test_set,
        batch_size=max(1, int(test_batch_size)),
        shuffle=False,
        num_workers=dataset_opt['dataloader_num_workers'],
        drop_last=False,
        pin_memory=True,
        persistent_workers=dataset_opt['dataloader_num_workers'] > 0,
        prefetch_factor=2 if dataset_opt['dataloader_num_workers'] > 0 else None,
    )


def _build_cwb_fallback_split(opt, seed):
    from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset

    train_opt = opt["datasets"]["train"]
    full_cfg = _build_cwb_cfg(train_opt, train_flag=True)
    full_cfg["all_times"] = True
    full_cfg["train"] = True
    full_set = get_zarr_dataset(**full_cfg)

    total = len(full_set)
    if total < 2:
        return None

    val_ratio = float(opt["train"].get("fallback_val_ratio", 0.1))
    val_ratio = min(max(val_ratio, 0.01), 0.5)
    val_size = max(1, int(round(total * val_ratio)))
    val_size = min(val_size, total - 1)

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(total, generator=gen).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_set = Subset(full_set, train_idx)
    test_set = Subset(full_set, val_idx)
    return train_set, test_set, total, len(train_idx), len(val_idx)


def _enforce_rain_single_output(opt):
    """Force true single-output rain training for gan_rain runs."""
    if str(opt.get("model", "")).lower() != "gan_rain":
        return opt

    train_opt = opt.setdefault("train", {})
    datasets_opt = opt.setdefault("datasets", {})
    netG_opt = opt.setdefault("netG", {})
    netD_opt = opt.setdefault("netD", {})

    train_ds = datasets_opt.get("train", {})
    test_ds = datasets_opt.get("test", {})

    # Allow a configurable multi-output mode while still using gan_rain + mask path.
    if not bool(train_opt.get("force_single_output", True)):
        configured_out = train_ds.get("out_channels", None)
        if isinstance(configured_out, (list, tuple)) and len(configured_out) > 0:
            target_out_channels = [int(v) for v in configured_out]
        else:
            target_out_channels = [0, 1, 2, 3]
        # Keep order, remove accidental duplicates.
        target_out_channels = list(dict.fromkeys(target_out_channels))

        for phase in ("train", "test"):
            ds = datasets_opt.get(phase, None)
            if isinstance(ds, dict):
                ds["out_channels"] = list(target_out_channels)

        n_out = len(target_out_channels)
        netG_opt["out_nc"] = n_out
        netD_opt["in_nc"] = n_out
        train_opt["train_only_tp_channel"] = False

        tp_idx = int(train_opt.get("tp_channel_idx", 0))
        tp_idx = max(0, min(tp_idx, n_out - 1))
        train_opt["tp_channel_idx"] = tp_idx
        train_opt["metric_channel_idx"] = int(
            train_opt.get("metric_channel_idx", tp_idx)
        )

        print(
            (
                "Multi-output mode enabled: "
                f"dataset out_channels={target_out_channels}, "
                f"netG.out_nc={n_out}, netD.in_nc={n_out}, "
                f"tp_channel_idx={train_opt['tp_channel_idx']}, "
                f"metric_channel_idx={train_opt['metric_channel_idx']}"
            ),
            flush=True,
        )
        return opt

    configured_out = train_ds.get("out_channels", None)
    configured_tp_idx = int(train_opt.get("tp_channel_idx", 0))
    detected = _detect_cwb_precip_channels(train_ds)

    if bool(train_opt.get("auto_detect_rain_channel", True)) and detected is not None:
        detected_out_idx = detected.get("cwb_precip_idx", None)
        detected_in_idx = detected.get("era5_precip_idx", None)
        era5_names = detected.get("era5_names", [])
        cwb_names = detected.get("cwb_names", [])

        if detected_out_idx is not None:
            rain_channel_id = int(detected_out_idx)
        else:
            rain_channel_id = None

        # For plotting LR precip-like channel: map absolute ERA5 index to selected input list index.
        if detected_in_idx is not None:
            in_sel = train_ds.get("in_channels", None)
            lr_plot_idx = None
            if isinstance(in_sel, (list, tuple)) and int(detected_in_idx) in in_sel:
                lr_plot_idx = int(in_sel.index(int(detected_in_idx)))
            elif isinstance(in_sel, (list, tuple)) and len(in_sel) > 0:
                lr_plot_idx = max(0, min(int(detected_in_idx), len(in_sel) - 1))
            else:
                lr_plot_idx = int(detected_in_idx)
            if int(train_opt.get("plot_lr_channel_idx", -1)) < 0:
                train_opt["plot_lr_channel_idx"] = int(lr_plot_idx)

        def _fmt(names):
            return ", ".join([f"{i}:{n}" for i, n in enumerate(names)])

        print(
            "Detected CWB channels: " + _fmt(cwb_names),
            flush=True,
        )
        print(
            "Detected ERA5 channels: " + _fmt(era5_names),
            flush=True,
        )
        if detected_out_idx is not None:
            print(
                f"Auto-detected HR precip channel idx={int(detected_out_idx)} name='{cwb_names[int(detected_out_idx)]}'",
                flush=True,
            )
        if detected_in_idx is not None:
            mapped = train_opt.get("plot_lr_channel_idx", None)
            era5_name = era5_names[int(detected_in_idx)] if int(detected_in_idx) < len(era5_names) else "unknown"
            print(
                f"Auto-detected LR precip-like channel idx={int(detected_in_idx)} name='{era5_name}', "
                f"plot_lr_channel_idx(mapped)={mapped}",
                flush=True,
            )
    else:
        rain_channel_id = None

    if rain_channel_id is None:
        if isinstance(configured_out, (list, tuple)) and len(configured_out) > 0:
            source_out_idx = max(0, min(configured_tp_idx, len(configured_out) - 1))
            rain_channel_id = int(configured_out[source_out_idx])
        else:
            rain_channel_id = max(0, configured_tp_idx)

    for phase in ("train", "test"):
        ds = datasets_opt.get(phase, None)
        if isinstance(ds, dict):
            ds["out_channels"] = [rain_channel_id]

    netG_opt["out_nc"] = 1
    netD_opt["in_nc"] = 1
    train_opt["train_only_tp_channel"] = True
    train_opt["tp_channel_idx"] = 0
    train_opt["metric_channel_idx"] = 0

    print(
        (
            "Rain-only single-output mode enabled: "
            f"dataset out_channels=[{rain_channel_id}], netG.out_nc=1, "
            "netD.in_nc=1, tp_channel_idx=0"
        ),
        flush=True,
    )
    return opt


def _decode_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _find_precip_idx(names, prefer_output=False):
    names_l = [n.lower() for n in names]
    # Priority order from precise to broad.
    if prefer_output:
        keywords = [
            "total_precipitation",
            "precipitation",
            "rain",
            "tp",
            "hourly_precip",
            "hourly_rain",
            "maximum_radar_reflectivity",
        ]
    else:
        keywords = [
            "total_precipitation",
            "precipitation",
            "rain",
            "tp",
            "hourly_precip",
            "hourly_rain",
            "precipitable_water",
        ]
    for kw in keywords:
        for i, n in enumerate(names_l):
            if kw in n:
                return i
    return None


def _detect_cwb_precip_channels(train_ds_opt):
    data_path = str(train_ds_opt.get("data_path", "") or "")
    if not data_path:
        return None


def _resolve_output_channel_labels(opt):
    labels = {}
    datasets_opt = opt.get("datasets", {}) if isinstance(opt, dict) else {}
    train_ds = datasets_opt.get("train", {}) if isinstance(datasets_opt, dict) else {}
    out_channels = train_ds.get("out_channels", []) if isinstance(train_ds, dict) else []
    if not isinstance(out_channels, (list, tuple)):
        out_channels = []

    detected = _detect_cwb_precip_channels(train_ds) if isinstance(train_ds, dict) else None
    cwb_names = detected.get("cwb_names", []) if isinstance(detected, dict) else []

    netg_out_nc = int(opt.get("netG", {}).get("out_nc", 1) or 1) if isinstance(opt, dict) else 1
    n = max(len(out_channels), netg_out_nc)
    for ch in range(n):
        abs_idx = out_channels[ch] if ch < len(out_channels) else ch
        name = None
        try:
            abs_i = int(abs_idx)
            if 0 <= abs_i < len(cwb_names):
                name = str(cwb_names[abs_i]).strip()
        except Exception:
            name = None
        if name:
            labels[ch] = f"ch{ch}_{name.replace(' ', '_')}"
        else:
            labels[ch] = f"ch{ch}"
    return labels


def _resolve_plot_channels(num_channels, metric_channel_idx, plot_all_channels=True):
    c = max(1, int(num_channels))
    if bool(plot_all_channels):
        return list(range(c))
    return [max(0, min(int(metric_channel_idx), c - 1))]
    try:
        group = zarr.open_consolidated(data_path)
        era5_names = [_decode_name(v) for v in group["era5_variable"][:]]
        cwb_names = [_decode_name(v) for v in group["cwb_variable"][:]]
        era5_precip_idx = _find_precip_idx(era5_names, prefer_output=False)
        cwb_precip_idx = _find_precip_idx(cwb_names, prefer_output=True)
        return {
            "era5_names": era5_names,
            "cwb_names": cwb_names,
            "era5_precip_idx": era5_precip_idx,
            "cwb_precip_idx": cwb_precip_idx,
        }
    except Exception as e:
        print(
            f"Warning: could not auto-detect precip channels from '{data_path}': {e}",
            flush=True,
        )
        return None


def _enable_rain_runtime_mask_fix(model, opt):
    """Rain-only runtime patch to avoid editing shared model files.

    - Replaces the internal mask scaling call so `weights` behaves like `mask`.
    - Adds a forward pre-hook that gates data channels with the UNet mask.
    """
    if str(opt.get("model", "")).lower() != "gan_rain":
        return

    bare_model = model.get_bare_model(model.netG)
    if not hasattr(bare_model, "unet") or not hasattr(bare_model, "sigmoid"):
        return

    class _MaskLinearizer(nn.Module):
        def forward(self, x):
            return x / 1000.0

    bare_model.sigmoid = _MaskLinearizer().to(model.device)

    def _rain_mask_pre_hook(module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            return None

        pos_channels = int(getattr(module, "pos_channels", 0) or 0)
        if pos_channels > 0:
            x_data = x[:, :-pos_channels, :, :]
            x_pos = x[:, -pos_channels:, :, :]
        else:
            x_data = x
            x_pos = None

        mask = module.unet(x_data)
        x_data = x_data * mask

        x_masked = torch.cat([x_data, x_pos], dim=1) if x_pos is not None else x_data
        return (x_masked,)

    handle = getattr(bare_model, "_rain_mask_hook_handle", None)
    if handle is not None:
        handle.remove()
    bare_model._rain_mask_hook_handle = bare_model.register_forward_pre_hook(
        _rain_mask_pre_hook
    )
    print("Rain-only runtime mask patch enabled.", flush=True)


def _select_metric_channel(img, channel_idx):
    if img.ndim != 3:
        return img
    c = img.shape[2]
    channel_idx = max(0, min(int(channel_idx), c - 1))
    return img[:, :, channel_idx : channel_idx + 1]


def _downsample_lr(x, scale):
    if scale <= 1:
        return x
    if x.dim() == 3:
        x = x.unsqueeze(0)
        lr_h = x.shape[-2] // scale
        lr_w = x.shape[-1] // scale
        x = F.interpolate(x, size=(lr_h, lr_w), mode="area")
        return x.squeeze(0)
    if x.dim() == 4:
        lr_h = x.shape[-2] // scale
        lr_w = x.shape[-1] // scale
        return F.interpolate(x, size=(lr_h, lr_w), mode="area")
    return x


def _prepare_model_batch(batch_data, scale, data_path):
    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
        img_clean = batch_data[0].float()
        img_lr = batch_data[1].float()
        img_lr = _downsample_lr(img_lr, scale) if scale > 1 else img_lr

        if img_clean.dim() == 3:
            img_clean = img_clean.unsqueeze(0)
        if img_lr.dim() == 3:
            img_lr = img_lr.unsqueeze(0)

        batch_size = int(img_lr.shape[0]) if img_lr.dim() == 4 else 1
        return {
            "L": img_lr,
            "H": img_clean,
            "L_path": [data_path] * batch_size,
            "H_path": [data_path] * batch_size,
        }
    return batch_data


def _log_internal_x_plots(
    model,
    dataset,
    scale,
    num_images,
    plots_dir,
    step,
    channel_idx=0,
):
    if num_images <= 0:
        return
    model.netG.eval()
    val_images = []
    val_predictions = []
    output_maps = []
    gt_maps = []

    with torch.no_grad():
        idx_candidates = [0, 98, 130, 236, 345, 443, 456, 654, 674, 879]
        idx_candidates = sorted(idx_candidates)[:num_images]
        max_idx = len(dataset)
        idx_candidates = [i for i in idx_candidates if i < max_idx]
        if not idx_candidates:
            model.netG.train()
            return
        for idx in idx_candidates:
            img_clean, img_lr = dataset[idx]
            img_lr = _downsample_lr(_as_tensor(img_lr), scale).unsqueeze(0).to(model.device)
            img_clean = _as_tensor(img_clean).unsqueeze(0).to(model.device)
            model.feed_data({"L": img_lr, "H": img_clean}, need_H=True)
            model.test()
            max_chan = int(model.E.shape[1]) - 1
            chan = max(0, min(int(channel_idx), max_chan))

            bare = model.get_bare_model(model.netG)
            internal_x = getattr(bare, "last_x", None)
            if internal_x is None:
                continue

            x_map = internal_x[0].mean(dim=0).detach().cpu()
            out_map = model.E.detach()[0, chan].cpu()

            val_images.append(out_map)
            val_predictions.append(x_map)
            output_maps.append(model.E.detach()[0, chan : chan + 1].cpu())
            gt_maps.append(model.H.detach()[0, chan : chan + 1].cpu())

    if not val_images:
        model.netG.train()
        return

    fig, axs = plt.subplots(2, len(val_images), figsize=(20, 8))
    for i in range(len(val_images)):
        vmin = min(val_images[i].min(), val_predictions[i].min())
        vmax = max(val_images[i].max(), val_predictions[i].max())
        axs[0, i].imshow(val_images[i].squeeze().numpy(), cmap="inferno", vmin=vmin, vmax=vmax)
        axs[1, i].imshow(val_predictions[i].squeeze().numpy(), cmap="inferno", vmin=vmin, vmax=vmax)
        axs[0, i].axis("off")
        axs[1, i].axis("off")

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"internal_x_step_{step}.png")
    plt.savefig(out_path, bbox_inches="tight")
    wandb.log({"internal_x": wandb.Image(fig)}, step=step)
    plt.close(fig)

    if output_maps and gt_maps:
        num_samples = len(output_maps)
        num_channels = 1
        ch_mins = []
        ch_maxs = []
        for c in range(num_channels):
            vals = []
            for i in range(num_samples):
                vals.append(output_maps[i][c])
                vals.append(gt_maps[i][c])
            ch_min = min(float(v.min()) for v in vals)
            ch_max = max(float(v.max()) for v in vals)
            ch_mins.append(ch_min)
            ch_maxs.append(ch_max)

        fig, axs = plt.subplots(num_channels, num_samples, figsize=(3 * num_samples, 3 * num_channels))
        if num_channels == 1 and num_samples == 1:
            axs = np.array([[axs]])
        elif num_channels == 1:
            axs = np.array([axs])
        elif num_samples == 1:
            axs = np.array([[ax] for ax in axs])

        for c in range(num_channels):
            for i in range(num_samples):
                axs[c, i].imshow(
                    output_maps[i][c].squeeze().numpy(),
                    cmap="inferno",
                    vmin=ch_mins[c],
                    vmax=ch_maxs[c],
                )
                axs[c, i].axis("off")

        out_path = os.path.join(plots_dir, f"outputs_step_{step}.png")
        plt.savefig(out_path, bbox_inches="tight")
        wandb.log({"outputs": wandb.Image(fig)}, step=step)
        plt.close(fig)

        fig, axs = plt.subplots(num_channels, num_samples, figsize=(3 * num_samples, 3 * num_channels))
        if num_channels == 1 and num_samples == 1:
            axs = np.array([[axs]])
        elif num_channels == 1:
            axs = np.array([axs])
        elif num_samples == 1:
            axs = np.array([[ax] for ax in axs])

        for c in range(num_channels):
            for i in range(num_samples):
                axs[c, i].imshow(
                    gt_maps[i][c].squeeze().numpy(),
                    cmap="inferno",
                    vmin=ch_mins[c],
                    vmax=ch_maxs[c],
                )
                axs[c, i].axis("off")

        out_path = os.path.join(plots_dir, f"gt_outputs_step_{step}.png")
        plt.savefig(out_path, bbox_inches="tight")
        wandb.log({"gt_outputs": wandb.Image(fig)}, step=step)
        plt.close(fig)

    model.netG.train()


def _collect_sample_panels(
    model,
    dataset,
    scale,
    sample_indices,
    channel_idx=0,
    lr_channel_idx=0,
):
    if dataset is None:
        return []
    sample_indices = [int(i) for i in sample_indices]
    sample_indices = [i for i in sample_indices if 0 <= i < len(dataset)]
    if not sample_indices:
        return []

    was_training = bool(model.netG.training)
    model.netG.eval()
    panels = []

    with torch.no_grad():
        for idx in sample_indices:
            img_clean, img_lr = dataset[idx]
            lr_tensor = _downsample_lr(_as_tensor(img_lr), scale)
            if lr_tensor.dim() == 2:
                lr_tensor = lr_tensor.unsqueeze(0)
            lr_ch = max(0, min(int(lr_channel_idx), int(lr_tensor.shape[0]) - 1))

            img_lr_batch = lr_tensor.unsqueeze(0).to(model.device)
            img_clean_t = _as_tensor(img_clean)
            if img_clean_t.dim() == 2:
                img_clean_t = img_clean_t.unsqueeze(0)
            img_clean_batch = img_clean_t.unsqueeze(0).to(model.device)

            model.feed_data({"L": img_lr_batch, "H": img_clean_batch}, need_H=True)
            model.test()

            max_chan = int(model.E.shape[1]) - 1
            chan = max(0, min(int(channel_idx), max_chan))

            pred = model.E.detach()[0, chan].float().cpu().numpy()
            gt = model.H.detach()[0, chan].float().cpu().numpy()
            err = np.abs(pred - gt)

            lr_ch_map = lr_tensor[lr_ch : lr_ch + 1].unsqueeze(0)
            lr_map = (
                F.interpolate(
                    lr_ch_map,
                    size=(pred.shape[0], pred.shape[1]),
                    mode="nearest",
                )[0, 0]
                .float()
                .cpu()
                .numpy()
            )

            mask_lr = model.mask.detach()[0:1].float().cpu()
            mask_hr = F.interpolate(
                mask_lr,
                size=(pred.shape[0], pred.shape[1]),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            panels.append(
                {
                    "idx": int(idx),
                    "lr": lr_map,
                    "gt": gt,
                    "pred": pred,
                    "mask": mask_hr,
                    "err": err,
                }
            )

    model.netG.train(was_training)
    return panels


def _save_index_grid(
    images,
    indices,
    title,
    out_path,
    cmap="magma",
    ncols=10,
    vmin=None,
    vmax=None,
):
    if not images:
        return None

    n = len(images)
    ncols = max(1, min(int(ncols), n))
    nrows = int(math.ceil(n / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(2.1 * ncols, 2.1 * nrows), constrained_layout=True
    )
    axs = np.array(axs, dtype=object).reshape(nrows, ncols)

    for k in range(nrows * ncols):
        r = k // ncols
        c = k % ncols
        ax = axs[r, c]
        if k < n:
            ax.imshow(images[k], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(f"idx {int(indices[k])}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_validation_selector_grids(
    model,
    dataset,
    scale,
    plots_dir,
    epoch_idx,
    channel_idx=0,
    channel_name=None,
    lr_channel_idx=0,
    n_samples=50,
    ncols=10,
    cmap="magma",
):
    if dataset is None or len(dataset) == 0:
        return {}

    total = len(dataset)
    n = max(1, min(int(n_samples), total))
    lin = np.linspace(0, total - 1, n, dtype=int).tolist()
    sample_indices = list(dict.fromkeys(lin))
    if len(sample_indices) < n:
        for i in range(total):
            if i not in sample_indices:
                sample_indices.append(i)
            if len(sample_indices) >= n:
                break

    panels = _collect_sample_panels(
        model=model,
        dataset=dataset,
        scale=scale,
        sample_indices=sample_indices,
        channel_idx=channel_idx,
        lr_channel_idx=lr_channel_idx,
    )
    if not panels:
        return {}

    os.makedirs(plots_dir, exist_ok=True)
    idxs = [p["idx"] for p in panels]
    lr_maps = [p["lr"] for p in panels]
    gt_maps = [p["gt"] for p in panels]
    pred_maps = [p["pred"] for p in panels]
    mask_maps = [p["mask"] for p in panels]

    precip_flat = np.concatenate([m.reshape(-1) for m in (gt_maps + pred_maps)])
    pmin = float(np.percentile(precip_flat, 1))
    pmax = float(np.percentile(precip_flat, 99))
    if pmax <= pmin:
        pmin, pmax = float(np.min(precip_flat)), float(np.max(precip_flat) + 1e-6)

    lr_flat = np.concatenate([m.reshape(-1) for m in lr_maps])
    lr_min = float(np.percentile(lr_flat, 1))
    lr_max = float(np.percentile(lr_flat, 99))
    if lr_max <= lr_min:
        lr_min, lr_max = float(np.min(lr_flat)), float(np.max(lr_flat) + 1e-6)

    out = {}
    channel_tag = f"ch{int(channel_idx)}"
    channel_name = str(channel_name) if channel_name else channel_tag
    out["lr"] = _save_index_grid(
        lr_maps,
        idxs,
        f"Validation Selector (epoch={int(epoch_idx)}) - LR channel {int(lr_channel_idx)} ({channel_name})",
        os.path.join(
            plots_dir,
            f"selector50_epoch_{int(epoch_idx):03d}_lr_ch{int(lr_channel_idx)}_{channel_tag}.png",
        ),
        cmap=cmap,
        ncols=ncols,
        vmin=lr_min,
        vmax=lr_max,
    )
    out["gt"] = _save_index_grid(
        gt_maps,
        idxs,
        f"Validation Selector (epoch={int(epoch_idx)}) - Ground Truth {channel_name}",
        os.path.join(plots_dir, f"selector50_epoch_{int(epoch_idx):03d}_gt_{channel_tag}.png"),
        cmap=cmap,
        ncols=ncols,
        vmin=pmin,
        vmax=pmax,
    )
    out["pred"] = _save_index_grid(
        pred_maps,
        idxs,
        f"Validation Selector (epoch={int(epoch_idx)}) - Output {channel_name}",
        os.path.join(plots_dir, f"selector50_epoch_{int(epoch_idx):03d}_pred_{channel_tag}.png"),
        cmap=cmap,
        ncols=ncols,
        vmin=pmin,
        vmax=pmax,
    )
    out["mask"] = _save_index_grid(
        mask_maps,
        idxs,
        f"Validation Selector (epoch={int(epoch_idx)}) - Feature Mask ({channel_name})",
        os.path.join(plots_dir, f"selector50_epoch_{int(epoch_idx):03d}_mask_{channel_tag}.png"),
        cmap=cmap,
        ncols=ncols,
        vmin=0.0,
        vmax=1.0,
    )

    payload = {"val_epoch": int(epoch_idx)}
    if out.get("lr"):
        payload[f"viz/selector/{channel_tag}/lr"] = wandb.Image(out["lr"])
    if out.get("gt"):
        payload[f"viz/selector/{channel_tag}/gt"] = wandb.Image(out["gt"])
    if out.get("pred"):
        payload[f"viz/selector/{channel_tag}/pred"] = wandb.Image(out["pred"])
    if out.get("mask"):
        payload[f"viz/selector/{channel_tag}/mask"] = wandb.Image(out["mask"])
    wandb.log(payload)
    return out


def _plot_epoch_gt_pred_mask(
    model,
    dataset,
    scale,
    plots_dir,
    epoch_idx,
    channel_idx=0,
    channel_name=None,
    lr_channel_idx=0,
    sample_indices=None,
    max_images=2,
    cmap="magma",
):
    if dataset is None:
        return None
    if sample_indices is None:
        sample_indices = [0, 98]
    sample_indices = [int(i) for i in sample_indices]
    sample_indices = [i for i in sample_indices if 0 <= i < len(dataset)]
    sample_indices = sample_indices[:max(1, int(max_images))]
    if not sample_indices:
        return None

    panels = _collect_sample_panels(
        model=model,
        dataset=dataset,
        scale=scale,
        sample_indices=sample_indices,
        channel_idx=channel_idx,
        lr_channel_idx=lr_channel_idx,
    )
    if not panels:
        return None

    os.makedirs(plots_dir, exist_ok=True)
    channel_tag = f"ch{int(channel_idx)}"
    channel_name = str(channel_name) if channel_name else channel_tag
    n = len(panels)
    fig, axs = plt.subplots(
        n, 5, figsize=(24, max(4.0, 4.4 * n)), constrained_layout=True
    )
    if n == 1:
        axs = np.array([axs])

    for i in range(n):
        panel = panels[i]
        idx = int(panel["idx"])
        lr = panel["lr"]
        gt = panel["gt"]
        pred = panel["pred"]
        mask = panel["mask"]
        err = panel["err"]

        pmin = float(min(pred.min(), gt.min()))
        pmax = float(max(pred.max(), gt.max()))
        if pmax <= pmin:
            pmax = pmin + 1e-6
        lr_min = float(lr.min())
        lr_max = float(lr.max())
        if lr_max <= lr_min:
            lr_max = lr_min + 1e-6
        err_max = float(err.max())
        if err_max <= 0:
            err_max = 1e-6

        im0 = axs[i, 0].imshow(
            lr, cmap=cmap, origin="lower", vmin=lr_min, vmax=lr_max
        )
        axs[i, 0].set_title(f"Sample {idx} - LR ch{int(lr_channel_idx)}")
        axs[i, 0].set_ylabel("Latitude")
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.046, pad=0.04)

        im1 = axs[i, 1].imshow(
            gt, cmap=cmap, origin="lower", vmin=pmin, vmax=pmax
        )
        axs[i, 1].set_title(f"Sample {idx} - Ground Truth {channel_name}")
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.046, pad=0.04)

        im2 = axs[i, 2].imshow(
            pred, cmap=cmap, origin="lower", vmin=pmin, vmax=pmax
        )
        axs[i, 2].set_title(f"Sample {idx} - Output {channel_name}")
        plt.colorbar(im2, ax=axs[i, 2], fraction=0.046, pad=0.04)

        im3 = axs[i, 3].imshow(mask, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0)
        axs[i, 3].set_title(f"Sample {idx} - Feature Mask")
        plt.colorbar(im3, ax=axs[i, 3], fraction=0.046, pad=0.04)

        im4 = axs[i, 4].imshow(err, cmap=cmap, origin="lower", vmin=0.0, vmax=err_max)
        axs[i, 4].set_title(f"Sample {idx} - |Output-GT|")
        plt.colorbar(im4, ax=axs[i, 4], fraction=0.046, pad=0.04)

    for c in range(5):
        axs[-1, c].set_xlabel("Longitude")

    fig.suptitle(
        f"Epoch {int(epoch_idx)} - Validation Panels ({channel_name})", fontsize=14
    )

    out_path = os.path.join(
        plots_dir, f"epoch_{int(epoch_idx):03d}_{channel_tag}_lr_gt_out_mask.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    wandb.log(
        {
            "val_epoch": int(epoch_idx),
            f"viz/epoch_panel/{channel_tag}": wandb.Image(out_path),
        }
    )
    plt.close(fig)
    return out_path


def main(json_path='../deep_learning/options/rainscaler_config_cwb_rain.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist

    opt = _apply_auto_paths(opt)
    opt = _apply_data_path_override(opt)
    opt = _enforce_rain_single_output(opt)


    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_D, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netD'] = init_path_D
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerD')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    opt['path']['pretrained_optimizerD'] = init_path_optimizerD
    current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)

    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    wandb_mode = str(opt["train"].get("wandb_mode", "online")).strip().lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        wandb_mode = "online"
    wandb.init(project="rainscale", mode=wandb_mode)
    wandb.define_metric("train_step")
    wandb.define_metric("G_loss", step_metric="train_step")
    wandb.define_metric("global_loss", step_metric="train_step")
    wandb.define_metric("M_loss", step_metric="train_step")
    wandb.define_metric("F_loss", step_metric="train_step")
    wandb.define_metric("D_loss", step_metric="train_step")
    wandb.define_metric("D_real", step_metric="train_step")
    wandb.define_metric("D_fake", step_metric="train_step")
    wandb.define_metric("val_epoch")
    wandb.define_metric("val/*", step_metric="val_epoch")
    wandb.define_metric("viz/*", step_metric="val_epoch")

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    train_batch_size = None
    train_sampler = None
    test_batch_size = 1
    test_set_is_empty = False
    train_dataset_type = str(opt["datasets"]["train"].get("dataset_type", "")).lower()
    test_dataset_type = str(opt["datasets"]["test"].get("dataset_type", "")).lower()
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_type = str(dataset_opt.get('dataset_type', '')).lower()
            if dataset_type in ['cwb', 'cwb_nemo']:
                from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
                train_cfg = _build_cwb_cfg(dataset_opt, train_flag=True)
                train_set = get_zarr_dataset(**train_cfg)
            else:
                train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader, train_sampler, train_batch_size = _make_train_loader(
                train_set, dataset_opt, opt, seed
            )

        elif phase == 'test':
            dataset_type = str(dataset_opt.get('dataset_type', '')).lower()
            if dataset_type in ['cwb', 'cwb_nemo']:
                from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
                test_cfg = _build_cwb_cfg(dataset_opt, train_flag=False)
                test_set = get_zarr_dataset(**test_cfg)
            else:
                test_set = define_Dataset(dataset_opt)
            test_batch_size = int(
                opt["train"].get(
                    "test_batch_size",
                    train_batch_size if train_batch_size is not None else 1,
                )
            )
            test_loader = _make_test_loader(test_set, dataset_opt, test_batch_size)
            print(
                f"Test steps per eval: {len(test_loader)} (batch_size={test_loader.batch_size})",
                flush=True,
            )
            if len(test_set) == 0:
                test_set_is_empty = True
                logger.warning(
                    "Validation dataset is empty after split. Validation will be skipped."
                )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if (
        test_set_is_empty
        and train_dataset_type in ['cwb', 'cwb_nemo']
        and test_dataset_type in ['cwb', 'cwb_nemo']
    ):
        fallback = _build_cwb_fallback_split(opt, seed)
        if fallback is not None:
            train_set, test_set, total, train_n, val_n = fallback
            train_loader, train_sampler, train_batch_size = _make_train_loader(
                train_set, opt["datasets"]["train"], opt, seed
            )
            test_loader = _make_test_loader(
                test_set, opt["datasets"]["test"], test_batch_size
            )
            test_set_is_empty = False
            logger.warning(
                "Year-based validation split was empty. Using fallback non-overlap "
                f"index split instead (total={total}, train={train_n}, val={val_n})."
            )
            print(
                f"Fallback CWB split active: train={train_n}, val={val_n}, "
                f"test steps per eval={len(test_loader)}",
                flush=True,
            )
        else:
            logger.warning(
                "Fallback split could not be created (dataset too small). "
                "Validation remains disabled."
            )

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    model.init_train()
    _enable_rain_runtime_mask_fix(model, opt)

   

    logger.info(model.info_network())
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    test_every_epochs = opt["train"].get("test_every_epochs", 0)
    metric_channel_idx = int(
        opt["train"].get("metric_channel_idx", opt["train"].get("tp_channel_idx", 0))
    )
    lr_plot_channel_idx = int(opt["train"].get("plot_lr_channel_idx", 0))
    plot_cmap = str(opt["train"].get("plot_cmap", "magma"))
    plot_all_channels = bool(opt["train"].get("plot_all_channels", True))
    val_all_channels = bool(opt["train"].get("val_all_channels", True))
    output_channel_labels = _resolve_output_channel_labels(opt)
    epochs = int(opt["train"].get("epochs", 100) or 100)
    test_step = int(opt["train"].get("test_start_step", 0) or 0)
    metric_eval_idx = int(metric_channel_idx)

    for epoch in range(epochs):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        epoch_batch_fetch_start = time.time()

        for i, train_data in enumerate(train_loader):
            if i == 0:
                first_batch_wait = time.time() - epoch_batch_fetch_start
                logger.info(
                    f"Epoch {epoch + 1}: first train batch fetch wait = {first_batch_wait:.2f}s"
                )

            current_step += 1
            print(f"Train Step is {current_step}")
            # -------------------------------
            # 1) optional internal_x plots
            # -------------------------------
            if (
                i == 0
                and opt['train'].get('plot_batch_internal_x', False)
            ):
                t_plot = time.time()
                num_images = int(opt['train'].get('plot_internal_x_n', 10))
                plot_dataset = test_set if 'test_set' in locals() else train_set
                plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
                _log_internal_x_plots(
                    model,
                    plot_dataset,
                    opt['scale'],
                    num_images,
                    plots_dir,
                    current_step,
                    metric_channel_idx,
                )
                logger.info(
                    f"Epoch {epoch + 1}: internal_x plotting took {time.time() - t_plot:.2f}s"
                )

            # -------------------------------
            # 2) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 3) feed patch pairs
            # -------------------------------
            train_data = _prepare_model_batch(
                train_data, opt["scale"], opt["datasets"]["train"]["data_path"]
            )
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            #TODO 
            model.optimize_parameters(current_step)



            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss


                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
                train_logs = {"train_step": current_step}
                train_logs.update(_sanitize_logs(logs))
                wandb.log(train_logs)
            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
            
            # -------------------------------
            # 6) testing
            # -------------------------------
            # TODO enable testing again
            # if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

            #     avg_psnr = 0.0
            #     avg_mae = 0.0
            #     avg_ssim = 0.0
            #     psnr_count = 0
            #     mae_count = 0
            #     ssim_count = 0
            #     idx = 0

            #     for test_data in test_loader:
            #         idx += 1
            #         if isinstance(test_data, (list, tuple)) and len(test_data) >= 2:
            #             img_clean = test_data[0].float()
            #             img_lr = test_data[1].float()
            #             if opt['scale'] > 1:
            #                 if img_lr.dim() == 3:
            #                     img_lr = img_lr.unsqueeze(0)
            #                     lr_h = img_lr.shape[-2] // opt['scale']
            #                     lr_w = img_lr.shape[-1] // opt['scale']
            #                     img_lr = F.interpolate(img_lr, size=(lr_h, lr_w), mode="area")
            #                     img_lr = img_lr.squeeze(0)
            #                 elif img_lr.dim() == 4:
            #                     lr_h = img_lr.shape[-2] // opt['scale']
            #                     lr_w = img_lr.shape[-1] // opt['scale']
            #                     img_lr = F.interpolate(img_lr, size=(lr_h, lr_w), mode="area")
            #             test_data = {
            #                 "L": img_lr,
            #                 "H": img_clean,
            #                 "L_path": [opt['datasets']['test']['data_path']],
            #                 "H_path": [opt['datasets']['test']['data_path']],
            #             }
            #         image_name_ext = os.path.basename(test_data['L_path'][0])
            #         img_name, ext = os.path.splitext(image_name_ext)
            #         # img_dir = opt['path']['images']
            #         # img_dir = os.path.join(opt['path']['images'], img_name)
            #         # util.mkdir(img_dir)

            #         model.feed_data(test_data)
            #         model.test()

            #         visuals = model.current_visuals()
            #         E_img = util.tensor2uint_regression(visuals['E'])
            #         H_img = util.tensor2uint_regression(visuals['H'])

            #         # -----------------------
            #         # save estimated image E
            #         # -----------------------
            #         # save_img_path = os.path.join(img_dir, '{:s}'.format(img_name))
            #         # save_img_path_p = os.path.join(img_dir, '{:s}.png'.format(img_name))
            #         #util.imsave(E_img * 140.0 * 255, save_img_path)
            #         # util.imsave_plt(E_img, save_img_path_p)
            #         # np.save(save_img_path,E_img)

            #        # -----------------------
            #         # calculate PSNR
            #         # -----------------------
            #         current_psnr,current_mae = util.calculate_score(E_img, H_img, border=border)

            #         current_ssim = util.calculate_ssim(E_img, H_img, border=border)
            #         psnr_str = "nan" if current_psnr is None else f"{current_psnr:<4.2f}"
            #         ssim_str = "nan" if current_ssim is None else f"{current_ssim:<7.5f}"
            #         mae_str  = "nan" if current_mae is None else f"{current_mae*100:<7.5f}"

            #         logger.info('{:->4d}--> {:>10s} | {}dB | {} | {} '.format(idx, image_name_ext, psnr_str, mae_str, ssim_str))

            #         wandb.log(
            #             {
            #                 "test_psnr_db": current_psnr,
            #                 "test_mae": current_mae * 100,
            #                 "test_ssim": current_ssim,
            #             },
            #             step=idx,
            #         )

            #         print(f"Test Step is:{idx}")

            #         if current_psnr is not None:
            #             avg_psnr += current_psnr
            #             psnr_count += 1
            #         if current_mae is not None:
            #             avg_mae += current_mae
            #             mae_count += 1
            #         if current_ssim is not None:
            #             avg_ssim += current_ssim
            #             ssim_count += 1
            #         avg_psnr = avg_psnr / psnr_count if psnr_count > 0 else float("nan")
            #         avg_mae  = avg_mae  / mae_count  if mae_count  > 0 else float("nan")
            #         avg_ssim = avg_ssim / ssim_count if ssim_count > 0 else float("nan")


            #     avg_psnr = avg_psnr / idx
            #     avg_mae = avg_mae / idx
            #     avg_ssim = avg_ssim / idx

            #     # testing log
            #     logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average MAE : {:<.5f} , Average SSIM : {:<.5f}\n'.format(epoch, current_step, avg_psnr, avg_mae*100, avg_ssim))

        # -------------------------------
        # 7) epoch-based testing (new)
        # -------------------------------
        if (
            opt["rank"] == 0
            and test_every_epochs
            and test_every_epochs > 0
            and "test_loader" in locals()
            and not test_set_is_empty
            and (epoch + 1) % test_every_epochs == 0
        ):
            t_val = time.time()
            test_step += 1
            logger.info(
                f"Starting validation at epoch {epoch + 1}/{epochs}, "
                f"test_step={test_step}, with {len(test_loader)} batches."
            )
            channel_sums = {}
            channel_counts = {}
            sample_idx = 0
            metric_eval_idx = int(metric_channel_idx)

            def _ensure_channel(ch):
                if ch not in channel_sums:
                    channel_sums[ch] = {
                        "psnr": 0.0,
                        "mae": 0.0,
                        "ssim": 0.0,
                        "mse": 0.0,
                        "rmse": 0.0,
                    }
                    channel_counts[ch] = {
                        "psnr": 0,
                        "mae": 0,
                        "ssim": 0,
                        "mse": 0,
                        "rmse": 0,
                    }

            def _safe_avg(total, count):
                return (total / count) if count > 0 else float("nan")

            val_log_each_sample = bool(opt["train"].get("val_log_each_sample", False))
            val_log_every_n = max(1, int(opt["train"].get("val_log_every_n", 100)))
            for batch_idx, test_data in enumerate(test_loader, start=1):
                test_data = _prepare_model_batch(
                    test_data, opt["scale"], opt["datasets"]["test"]["data_path"]
                )
                image_name_ext = os.path.basename(test_data["L_path"][0])
                model.feed_data(test_data)
                model.test()

                E_batch = model.E.detach().float().cpu()
                H_batch = model.H.detach().float().cpu()
                if E_batch.dim() == 3:
                    E_batch = E_batch.unsqueeze(0)
                if H_batch.dim() == 3:
                    H_batch = H_batch.unsqueeze(0)

                for b in range(E_batch.shape[0]):
                    sample_idx += 1
                    E_img = util.tensor2uint_regression(E_batch[b])
                    H_img = util.tensor2uint_regression(H_batch[b])
                    n_eval_channels = int(E_img.shape[2]) if E_img.ndim == 3 else 1
                    eval_channels = _resolve_plot_channels(
                        n_eval_channels,
                        metric_channel_idx,
                        plot_all_channels=val_all_channels,
                    )
                    metric_eval_idx = max(0, min(int(metric_channel_idx), n_eval_channels - 1))

                    current_psnr = None
                    current_mae = None
                    current_ssim = None
                    current_mse = None
                    current_rmse = None

                    for ch in eval_channels:
                        _ensure_channel(ch)
                        E_metric = _select_metric_channel(E_img, ch)
                        H_metric = _select_metric_channel(H_img, ch)
                        ch_psnr, ch_mae = util.calculate_score(
                            E_metric, H_metric, border=border
                        )
                        ch_ssim = util.calculate_ssim(E_metric, H_metric, border=border)
                        diff = E_metric.astype(np.float64) - H_metric.astype(np.float64)
                        ch_mse = float(np.mean(diff ** 2))
                        ch_rmse = float(np.sqrt(ch_mse))

                        if ch_psnr is not None:
                            channel_sums[ch]["psnr"] += ch_psnr
                            channel_counts[ch]["psnr"] += 1
                        if ch_mae is not None:
                            channel_sums[ch]["mae"] += ch_mae
                            channel_counts[ch]["mae"] += 1
                        if ch_ssim is not None:
                            channel_sums[ch]["ssim"] += ch_ssim
                            channel_counts[ch]["ssim"] += 1
                        channel_sums[ch]["mse"] += ch_mse
                        channel_sums[ch]["rmse"] += ch_rmse
                        channel_counts[ch]["mse"] += 1
                        channel_counts[ch]["rmse"] += 1

                        if ch == metric_eval_idx:
                            current_psnr = ch_psnr
                            current_mae = ch_mae
                            current_ssim = ch_ssim
                            current_mse = ch_mse
                            current_rmse = ch_rmse

                    if current_mse is None:
                        current_mse = float("nan")
                    if current_rmse is None:
                        current_rmse = float("nan")
                    psnr_str = "nan" if current_psnr is None else f"{current_psnr:<4.2f}"
                    ssim_str = "nan" if current_ssim is None else f"{current_ssim:<7.5f}"
                    mae_str = "nan" if current_mae is None else f"{current_mae*100:<7.5f}"

                    if val_log_each_sample and (
                        sample_idx % val_log_every_n == 0 or sample_idx == 1
                    ):
                        logger.info(
                            "{:->4d}--> {:>10s} | {}dB | {} | {} ".format(
                                sample_idx, image_name_ext, psnr_str, mae_str, ssim_str
                            )
                        )

                if batch_idx % 20 == 0 or batch_idx == len(test_loader):
                    logger.info(
                        f"Validation progress: batch {batch_idx}/{len(test_loader)}"
                    )

            _ensure_channel(metric_eval_idx)
            avg_psnr = _safe_avg(
                channel_sums[metric_eval_idx]["psnr"],
                channel_counts[metric_eval_idx]["psnr"],
            )
            avg_mae = _safe_avg(
                channel_sums[metric_eval_idx]["mae"],
                channel_counts[metric_eval_idx]["mae"],
            )
            avg_ssim = _safe_avg(
                channel_sums[metric_eval_idx]["ssim"],
                channel_counts[metric_eval_idx]["ssim"],
            )
            avg_mse = _safe_avg(
                channel_sums[metric_eval_idx]["mse"],
                channel_counts[metric_eval_idx]["mse"],
            )
            avg_rmse = _safe_avg(
                channel_sums[metric_eval_idx]["rmse"],
                channel_counts[metric_eval_idx]["rmse"],
            )

            logger.info(
                "<epoch:{:3d}, iter:{:8,d}, test_step:{:5d}, eval_samples:{:6d}, Average PSNR : {:<.2f}dB, Average MAE : {:<.5f} , Average SSIM : {:<.5f}, Average MSE : {:<.6f}, Average RMSE : {:<.6f}\n".format(
                    epoch + 1, current_step, test_step, sample_idx, avg_psnr, avg_mae * 100, avg_ssim, avg_mse, avg_rmse
                )
            )
            val_payload = {
                "val_epoch": int(epoch),
                "train_step": current_step,
                "val/psnr_db": avg_psnr,
                "val/mae": avg_mae * 100 if avg_mae == avg_mae else avg_mae,
                "val/ssim": avg_ssim,
                "val/mse": avg_mse,
                "val/rmse": avg_rmse,
            }
            for ch in sorted(channel_sums.keys()):
                ch_psnr = _safe_avg(channel_sums[ch]["psnr"], channel_counts[ch]["psnr"])
                ch_mae = _safe_avg(channel_sums[ch]["mae"], channel_counts[ch]["mae"])
                ch_ssim = _safe_avg(channel_sums[ch]["ssim"], channel_counts[ch]["ssim"])
                ch_mse = _safe_avg(channel_sums[ch]["mse"], channel_counts[ch]["mse"])
                ch_rmse = _safe_avg(channel_sums[ch]["rmse"], channel_counts[ch]["rmse"])
                val_payload[f"val/ch{ch}/psnr_db"] = ch_psnr
                val_payload[f"val/ch{ch}/mae"] = ch_mae * 100 if ch_mae == ch_mae else ch_mae
                val_payload[f"val/ch{ch}/ssim"] = ch_ssim
                val_payload[f"val/ch{ch}/mse"] = ch_mse
                val_payload[f"val/ch{ch}/rmse"] = ch_rmse
            wandb.log(val_payload)
            logger.info(
                f"Validation wall time (epoch {epoch + 1}) = {time.time() - t_val:.2f}s"
            )

            selector_every_epoch = bool(
                opt["train"].get("plot_selector_every_epoch", False)
            )
            selector_once = bool(opt["train"].get("plot_selector_once", True))
            run_selector = selector_every_epoch or (selector_once and int(epoch) == 0)

            if opt["rank"] == 0 and run_selector:
                t_selector = time.time()
                selector_n = int(opt["train"].get("plot_selector_n", 50))
                selector_cols = int(opt["train"].get("plot_selector_cols", 10))
                plots_dir = os.path.join(os.path.dirname(__file__), "plots")
                try:
                    num_out_channels = int(model.E.shape[1]) if hasattr(model, "E") else 1
                    plot_channels = _resolve_plot_channels(
                        num_out_channels,
                        metric_eval_idx,
                        plot_all_channels=plot_all_channels,
                    )
                    for ch in plot_channels:
                        ch_name = output_channel_labels.get(ch, f"ch{ch}")
                        selector_paths = _plot_validation_selector_grids(
                            model=model,
                            dataset=test_set if "test_set" in locals() else train_set,
                            scale=opt["scale"],
                            plots_dir=plots_dir,
                            epoch_idx=epoch,
                            channel_idx=ch,
                            channel_name=ch_name,
                            lr_channel_idx=lr_plot_channel_idx,
                            n_samples=selector_n,
                            ncols=selector_cols,
                            cmap=plot_cmap,
                        )
                        for k, v in selector_paths.items():
                            if v:
                                logger.info(f"Selector ch{ch} ({ch_name}) {k}: {v}")
                    logger.info(
                        f"Epoch {epoch + 1}: selector plotting took {time.time() - t_selector:.2f}s"
                    )
                except Exception as e:
                    logger.exception(
                        f"Epoch {epoch + 1}: selector plotting failed: {e}"
                    )

        if opt["rank"] == 0 and opt["train"].get("plot_epoch_samples", True):
            t_plot_epoch = time.time()
            plot_dataset = test_set if 'test_set' in locals() else train_set
            sample_indices = opt["train"].get(
                "plot_sample_indices",
                [0, 98],
            )
            max_images = int(opt["train"].get("plot_epoch_n", 2))
            plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
            try:
                num_out_channels = int(model.E.shape[1]) if hasattr(model, "E") else 1
                plot_channels = _resolve_plot_channels(
                    num_out_channels,
                    metric_eval_idx,
                    plot_all_channels=plot_all_channels,
                )
                for ch in plot_channels:
                    ch_name = output_channel_labels.get(ch, f"ch{ch}")
                    saved_plot_path = _plot_epoch_gt_pred_mask(
                        model=model,
                        dataset=plot_dataset,
                        scale=opt["scale"],
                        plots_dir=plots_dir,
                        epoch_idx=epoch,
                        channel_idx=ch,
                        channel_name=ch_name,
                        lr_channel_idx=lr_plot_channel_idx,
                        sample_indices=sample_indices,
                        max_images=max_images,
                        cmap=plot_cmap,
                    )
                    if saved_plot_path:
                        logger.info(
                            f"Epoch {epoch + 1}: saved epoch plot ch{ch} ({ch_name}) to {saved_plot_path}"
                        )
                logger.info(
                    f"Epoch {epoch + 1}: epoch sample plotting took {time.time() - t_plot_epoch:.2f}s"
                )
            except Exception as e:
                logger.exception(
                    f"Epoch {epoch + 1}: epoch sample plotting failed: {e}"
                )

if __name__ == '__main__':
    main()
