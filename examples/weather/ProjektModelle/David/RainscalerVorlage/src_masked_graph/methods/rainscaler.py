import os.path
import sys
sys.path.insert(0, '../deep_learning')
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import wandb


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

    wandb_run = None
    if opt['rank'] == 0:
        wandb_run = wandb.init(
            project="rainscale",
            name=opt["task"] if "task" in opt else None,
            config=option.nonedict_to_dict(opt) if hasattr(option, "nonedict_to_dict") else dict(opt),
        )



    


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
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
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

    for epoch in range(10):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        

        for i, train_data in enumerate(train_loader):

            current_step += 1
            
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
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

                if wandb_run is not None:
                    payload = {}
                    for k, v in logs.items():
                        payload[f"train/{k}"] = float(v)
                    wandb.log(payload, step=int(current_step))


            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
            
            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_mae = 0.0
                avg_ssim = 0.0
                psnr_count = 0
                mae_count = 0
                ssim_count = 0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)
                    # img_dir = opt['path']['images']
                    # img_dir = os.path.join(opt['path']['images'], img_name)
                    # util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.testx8()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint_regression(visuals['E'])
                    H_img = util.tensor2uint_regression(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    # save_img_path = os.path.join(img_dir, '{:s}'.format(img_name))
                    # save_img_path_p = os.path.join(img_dir, '{:s}.png'.format(img_name))
                    #util.imsave(E_img * 140.0 * 255, save_img_path)
                    # util.imsave_plt(E_img, save_img_path_p)
                    # np.save(save_img_path,E_img)

                   # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr,current_mae = util.calculate_score(E_img, H_img, border=border)

                    if wandb_run is not None:
                        if current_psnr is not None and np.isfinite(current_psnr):
                            wandb.log(
                                {
                                    "psnr_db": float(current_psnr),
                                    "mae": float(current_mae * 100),
                                },
                                step=int(current_step),
                            )

                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    psnr_str = "nan" if current_psnr is None else f"{current_psnr:<4.2f}"
                    ssim_str = "nan" if current_ssim is None else f"{current_ssim:<7.5f}"
                    mae_str  = "nan" if current_mae is None else f"{current_mae*100:<7.5f}"

                    logger.info('{:->4d}--> {:>10s} | {}dB | {} | {} '.format(idx, image_name_ext, psnr_str, mae_str, ssim_str))

                    if wandb_run is not None:
                        payload = {}
                        if current_psnr is not None and np.isfinite(current_psnr):
                            payload["test/psnr_db"] = float(current_psnr)
                        if current_mae is not None and np.isfinite(current_mae):
                            payload["test/mae"] = float(current_mae * 100) 
                        if payload:
                            wandb.log(payload, step=int(current_step))



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
                    avg_mae  = avg_mae  / mae_count  if mae_count  > 0 else float("nan")
                    avg_ssim = avg_ssim / ssim_count if ssim_count > 0 else float("nan")


                avg_psnr = avg_psnr / idx
                avg_mae = avg_mae / idx
                avg_ssim = avg_ssim / idx

                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average MAE : {:<.5f} , Average SSIM : {:<.5f}\n'.format(epoch, current_step, avg_psnr, avg_mae*100, avg_ssim))

    if wandb_run is not None:
        wandb.finish()

if __name__ == '__main__':
    main()