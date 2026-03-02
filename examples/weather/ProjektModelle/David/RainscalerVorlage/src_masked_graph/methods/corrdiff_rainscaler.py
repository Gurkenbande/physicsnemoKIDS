import os
import sys
import math
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, "../deep_learning")

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from models.select_model import define_Model


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


_add_physicsnemo_root()
_add_corrdiff_root()

from datasets.dataset import init_train_valid_datasets_from_config
from physicsnemo.distributed import DistributedManager

import wandb


def _build_cwb_cfg(dataset_opt, train_flag):
    cfg = {
        "type": "cwb",
        "data_path": dataset_opt["data_path"],
        "normalization": dataset_opt.get("normalization", "v1"),
        "all_times": dataset_opt.get("all_times", False),
        "train": train_flag,
        "in_channels": dataset_opt.get("in_channels", None),
        "out_channels": dataset_opt.get("out_channels", None),
        "img_shape_x": dataset_opt.get("img_shape_x", 448),
        "img_shape_y": dataset_opt.get("img_shape_y", 448),
        "add_grid": dataset_opt.get("add_grid", True),
        "ds_factor": dataset_opt.get("ds_factor", 4),
        "n_history": dataset_opt.get("n_history", 0),
        "min_path": dataset_opt.get("min_path", None),
        "max_path": dataset_opt.get("max_path", None),
        "global_means_path": dataset_opt.get("global_means_path", None),
        "global_stds_path": dataset_opt.get("global_stds_path", None),
    }
    return {k: v for k, v in cfg.items() if v is not None}


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


def _log_internal_x_plots(
    model,
    dataset,
    scale,
    num_images,
    plots_dir,
    step,
):
    if num_images <= 0:
        return
    model.netG.eval()
    val_images = []
    val_predictions = []
    output_maps = []
    gt_maps = []

    with torch.no_grad():
        for idx in range(num_images):
            img_clean, img_lr = dataset[idx]
            img_lr = _downsample_lr(_as_tensor(img_lr), scale).unsqueeze(0).to(model.device)
            img_clean = _as_tensor(img_clean).unsqueeze(0).to(model.device)
            model.feed_data({"L": img_lr, "H": img_clean}, need_H=True)
            model.test()

            bare = model.get_bare_model(model.netG)
            internal_x = getattr(bare, "last_x", None)
            if internal_x is None:
                continue

            x_map = internal_x[0].mean(dim=0).detach().cpu()
            if model.E.shape[1] > 3:
                out_map = model.E.detach()[0, 3].cpu()
            else:
                out_map = model.E.detach()[0, 0].cpu()

            val_images.append(out_map)
            val_predictions.append(x_map)
            output_maps.append(model.E.detach()[0].cpu())
            gt_maps.append(model.H.detach()[0].cpu())

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
        num_channels = min(4, output_maps[0].shape[0])
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


def main(json_path="../deep_learning/options/rainscaler_config.json"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path, help="Path to option JSON file.")
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dist", default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt["dist"] = parser.parse_args().dist

    DistributedManager.initialize()

    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()

    if opt["rank"] == 0:
        util.mkdirs((path for key, path in opt["path"].items() if "pretrained" not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt["path"]["models"], net_type="G")
    init_iter_D, init_path_D = option.find_last_checkpoint(opt["path"]["models"], net_type="D")
    init_iter_E, init_path_E = option.find_last_checkpoint(opt["path"]["models"], net_type="E")
    opt["path"]["pretrained_netG"] = init_path_G
    opt["path"]["pretrained_netD"] = init_path_D
    opt["path"]["pretrained_netE"] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt["path"]["models"], net_type="optimizerG")
    init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt["path"]["models"], net_type="optimizerD")
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    opt["path"]["pretrained_optimizerD"] = init_path_optimizerD
    current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)
    border = opt["scale"]

    if opt["rank"] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt["rank"] == 0:
        logger_name = "train"
        utils_logger.logger_info(logger_name, os.path.join(opt["path"]["log"], logger_name + ".log"))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    wandb.init(project="rainscale")

    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_opt = opt["datasets"]["train"]
    test_opt = opt["datasets"].get("test", None)
    batch_size = train_opt["dataloader_batch_size"]
    num_workers = train_opt.get("dataloader_num_workers", 0)
    manager = DistributedManager()
    total_batch_size = batch_size * max(1, manager.world_size)

    dataloader_kwargs = {
        "pin_memory": True,
        "num_workers": num_workers,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }

    train_cfg = _build_cwb_cfg(train_opt, train_flag=True)
    test_cfg = _build_cwb_cfg(test_opt, train_flag=False) if test_opt else None

    (
        train_set,
        train_iter,
        test_set,
        test_iter,
    ) = init_train_valid_datasets_from_config(
        train_cfg,
        dataloader_kwargs,
        batch_size=batch_size,
        seed=seed,
        validation_dataset_cfg=test_cfg,
        validation=bool(test_opt),
        sampler_start_idx=current_step * total_batch_size,
    )

    train_steps_per_epoch = int(math.ceil(len(train_set) / batch_size))
    print(train_steps_per_epoch)
    test_steps = int(math.ceil(len(test_set) / test_opt["dataloader_batch_size"])) if test_opt else 0

    if opt["rank"] == 0:
        logger.info(
            "Number of train images: {:,d}, iters: {:,d}".format(len(train_set), train_steps_per_epoch)
        )

    model = define_Model(opt)
    model.init_train()

    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    epochs = opt["train"].get("epochs", 10) or 10

    cur_nimg = current_step * total_batch_size

    for epoch in range(epochs):
        for i in range(train_steps_per_epoch):
            current_step += 1
            print(current_step, flush=True)
            cur_nimg += total_batch_size

            if (
                opt["rank"] == 0
                and i == 0
                and opt["train"].get("plot_internal_x", False)
            ):
                num_images = int(opt["train"].get("plot_internal_x_n", 10))
                plot_dataset = test_set if test_opt else train_set
                plots_dir = os.path.join(os.path.dirname(__file__), "plots")
                _log_internal_x_plots(
                    model,
                    plot_dataset,
                    opt["scale"],
                    num_images,
                    plots_dir,
                    cur_nimg,
                )

            img_clean, img_lr, *lead_time = next(train_iter)
            img_lr = _downsample_lr(_as_tensor(img_lr), opt["scale"])
            img_clean = _as_tensor(img_clean)
            train_data = {
                "L": img_lr,
                "H": img_clean,
                "L_path": [train_opt["data_path"]],
                "H_path": [train_opt["data_path"]],
            }

            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if opt["rank"] == 0:
                logs = _sanitize_logs(model.current_log())
                wandb.log({f"train/{k}": v for k, v in logs.items()}, step=cur_nimg)

            if current_step % opt["train"]["checkpoint_print"] == 0 and opt["rank"] == 0:
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.3e} ".format(k, v)
                logger.info(message)

            if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
                logger.info("Saving the model.")
                model.save(current_step)

            if (
                test_opt
                and current_step % opt["train"]["checkpoint_test"] == 0
                and opt["rank"] == 0
            ):
                avg_psnr = 0.0
                avg_mae = 0.0
                avg_ssim = 0.0
                psnr_count = 0
                mae_count = 0
                ssim_count = 0

                for idx in range(test_steps):
                    img_clean, img_lr, *lead_time = next(test_iter)
                    img_lr = _downsample_lr(_as_tensor(img_lr), opt["scale"])
                    img_clean = _as_tensor(img_clean)
                    test_data = {
                        "L": img_lr,
                        "H": img_clean,
                        "L_path": [test_opt["data_path"]],
                        "H_path": [test_opt["data_path"]],
                    }

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint_regression(visuals["E"])
                    H_img = util.tensor2uint_regression(visuals["H"])

                    current_psnr, current_mae = util.calculate_score(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    psnr_str = "nan" if current_psnr is None else f"{current_psnr:<4.2f}"
                    ssim_str = "nan" if current_ssim is None else f"{current_ssim:<7.5f}"
                    mae_str = "nan" if current_mae is None else f"{current_mae*100:<7.5f}"

                    logger.info(
                        "{:->4d}--> {:>10s} | {}dB | {} | {} ".format(
                            idx + 1, os.path.basename(test_opt["data_path"]), psnr_str, mae_str, ssim_str
                        )
                    )

                    if current_psnr is not None:
                        avg_psnr += current_psnr
                        psnr_count += 1
                    if current_mae is not None:
                        avg_mae += current_mae
                        mae_count += 1
                    if current_ssim is not None:
                        avg_ssim += current_ssim
                        ssim_count += 1

                avg_psnr = avg_psnr / psnr_count if psnr_count > 0 else float("nan")
                avg_mae = avg_mae / mae_count if mae_count > 0 else float("nan")
                avg_ssim = avg_ssim / ssim_count if ssim_count > 0 else float("nan")

                logger.info(
                    "<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average MAE : {:<.5f} , Average SSIM : {:<.5f}\n".format(
                        epoch, current_step, avg_psnr, avg_mae * 100, avg_ssim
                    )
                )
                wandb.log(
                    {
                        "test/psnr_db": avg_psnr,
                        "test/mae": avg_mae * 100,
                        "test/ssim": avg_ssim,
                    },
                    step=current_step,
                )


if __name__ == "__main__":
    main()
