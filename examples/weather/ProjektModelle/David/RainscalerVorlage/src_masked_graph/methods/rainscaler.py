import os.path
import sys
import os
sys.path.insert(0, '../deep_learning')
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

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


def main(json_path='../deep_learning/options/rainscaler_config.json'):

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

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    opt = _apply_auto_paths(opt)
    opt = _apply_data_path_override(opt)


    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
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
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    if opt['rank'] == 0:
        wandb.init(project="rainscale")

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
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_type = str(dataset_opt.get('dataset_type', '')).lower()
            if dataset_type in ['cwb', 'cwb_nemo']:
                from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
                data_path = Path(dataset_opt['data_path']).expanduser()
                train_set = get_zarr_dataset(data_path=data_path)
            else:
                train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          persistent_workers=(dataset_opt['dataloader_num_workers']//opt['num_gpu']) > 0,
                                          prefetch_factor=2 if (dataset_opt['dataloader_num_workers']//opt['num_gpu']) > 0 else None,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True,
                                          persistent_workers=dataset_opt['dataloader_num_workers'] > 0,
                                          prefetch_factor=2 if dataset_opt['dataloader_num_workers'] > 0 else None)

        elif phase == 'test':
            dataset_type = str(dataset_opt.get('dataset_type', '')).lower()
            if dataset_type in ['cwb', 'cwb_nemo']:
                from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
                data_path = Path(dataset_opt['data_path']).expanduser()
                test_set = get_zarr_dataset(data_path=data_path)
            else:
                test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=dataset_opt['dataloader_num_workers'],
                                     drop_last=False, pin_memory=True,
                                     persistent_workers=dataset_opt['dataloader_num_workers'] > 0,
                                     prefetch_factor=2 if dataset_opt['dataloader_num_workers'] > 0 else None)
            print(f"Test steps per eval: {len(test_loader)}", flush=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)

    model.init_train()

   

    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    test_every_epochs = opt["train"].get("test_every_epochs", 0)

    for epoch in range(30):  # keep running #might do less epochs idk
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        

        for i, train_data in enumerate(train_loader):

            current_step += 1
            print(f"Train Step is {current_step}")
            # -------------------------------
            # 1) optional internal_x plots
            # -------------------------------
            if (
                opt['rank'] == 0
                and i == 0
                and opt['train'].get('plot_internal_x', False)
            ):
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
                )

            # -------------------------------
            # 2) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 3) feed patch pairs
            # -------------------------------
            if isinstance(train_data, (list, tuple)) and len(train_data) >= 2:
                img_clean = train_data[0].float()
                img_lr = train_data[1].float()
                if opt['scale'] > 1:
                    if img_lr.dim() == 3:
                        img_lr = img_lr.unsqueeze(0)
                        lr_h = img_lr.shape[-2] // opt['scale']
                        lr_w = img_lr.shape[-1] // opt['scale']
                        img_lr = F.interpolate(img_lr, size=(lr_h, lr_w), mode="area")
                        img_lr = img_lr.squeeze(0)
                    elif img_lr.dim() == 4:
                        lr_h = img_lr.shape[-2] // opt['scale']
                        lr_w = img_lr.shape[-1] // opt['scale']
                        img_lr = F.interpolate(img_lr, size=(lr_h, lr_w), mode="area")
                train_data = {
                    "L": img_lr,
                    "H": img_clean,
                    "L_path": [opt['datasets']['train']['data_path']],
                    "H_path": [opt['datasets']['train']['data_path']],
                }
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            #TODO 
            model.optimize_parameters(current_step)



            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss


                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

                wandb.log(logs, step=current_step)
            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
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
            and (epoch + 1) % test_every_epochs == 0
        ):
            avg_psnr = 0.0
            avg_mae = 0.0
            avg_ssim = 0.0
            psnr_count = 0
            mae_count = 0
            ssim_count = 0
            idx = 0

            for test_data in test_loader:
                idx += 1
                if isinstance(test_data, (list, tuple)) and len(test_data) >= 2:
                    img_clean = test_data[0].float()
                    img_lr = test_data[1].float()
                    if opt["scale"] > 1:
                        if img_lr.dim() == 3:
                            img_lr = img_lr.unsqueeze(0)
                            lr_h = img_lr.shape[-2] // opt["scale"]
                            lr_w = img_lr.shape[-1] // opt["scale"]
                            img_lr = F.interpolate(
                                img_lr, size=(lr_h, lr_w), mode="area"
                            )
                            img_lr = img_lr.squeeze(0)
                        elif img_lr.dim() == 4:
                            lr_h = img_lr.shape[-2] // opt["scale"]
                            lr_w = img_lr.shape[-1] // opt["scale"]
                            img_lr = F.interpolate(
                                img_lr, size=(lr_h, lr_w), mode="area"
                            )
                    test_data = {
                        "L": img_lr,
                        "H": img_clean,
                        "L_path": [opt["datasets"]["test"]["data_path"]],
                        "H_path": [opt["datasets"]["test"]["data_path"]],
                    }
                image_name_ext = os.path.basename(test_data["L_path"][0])

                model.feed_data(test_data)
                model.test()

                visuals = model.current_visuals()
                E_img = util.tensor2uint_regression(visuals["E"])
                H_img = util.tensor2uint_regression(visuals["H"])

                current_psnr, current_mae = util.calculate_score(
                    E_img, H_img, border=border
                )
                current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                psnr_str = "nan" if current_psnr is None else f"{current_psnr:<4.2f}"
                ssim_str = "nan" if current_ssim is None else f"{current_ssim:<7.5f}"
                mae_str = "nan" if current_mae is None else f"{current_mae*100:<7.5f}"

                logger.info(
                    "{:->4d}--> {:>10s} | {}dB | {} | {} ".format(
                        idx, image_name_ext, psnr_str, mae_str, ssim_str
                    )
                )

                wandb.log(
                    {
                        "test_psnr_db": current_psnr,
                        "test_mae": None if current_mae is None else current_mae * 100,
                        "test_ssim": current_ssim,
                    },
                    step=idx,
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

if __name__ == '__main__':
    main()
