# ------------------------------------------------------------------
#  Train script  - HRNet Face Alignment
#  修改：统一 device，CPU 环境不再报错
# ------------------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os, sys, pprint, argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment (HRNet)')
    parser.add_argument('--cfg', required=True, type=str,
                        help='experiment yaml config')
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # ------------- CUDA / CPU -------------
    gpus     = list(config.GPUS)
    cuda_ok  = torch.cuda.is_available() and len(gpus)
    device   = torch.device('cuda' if cuda_ok else 'cpu')

    if cuda_ok:
        print('=> Using CUDA on GPU(s):', gpus)
    else:
        print('=> running on *CPU* only')

    cudnn.benchmark     = config.CUDNN.BENCHMARK and cuda_ok
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled       = config.CUDNN.ENABLED and cuda_ok

    # ------------- model -------------
    model = models.get_face_alignment_net(config)
    if cuda_ok:
        model = nn.DataParallel(model, device_ids=gpus).to(device)
    else:
        model = model.to(device)

    # ------------- loss & optimizer -------------
    criterion = torch.nn.MSELoss(reduction='mean').to(device)
    optimizer = utils.get_optimizer(config, model)

    # ------------- tensorboard -------------
    writer_dict = {
        'writer'             : SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps' : 0,
        'valid_global_steps' : 0,
    }

    # ------------- resume -------------
    best_nme   = 1e8
    last_epoch = config.TRAIN.BEGIN_EPOCH
    ckpt_file  = os.path.join(final_output_dir, 'latest.pth')

    if config.TRAIN.RESUME and os.path.isfile(ckpt_file):
        ckpt       = torch.load(ckpt_file, map_location=device)
        last_epoch = ckpt['epoch']
        best_nme   = ckpt['best_nme']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f'=> resume checkpoint (epoch {ckpt["epoch"]})')
    else:
        print('=> no checkpoint found')

    # ------------- LR scheduler -------------
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch-1)

    # ------------- dataset / dataloader -------------
    DatasetCls = get_dataset(config)

    train_loader = DataLoader(
        DatasetCls(config, is_train=True),
        batch_size=max(1, config.TRAIN.BATCH_SIZE_PER_GPU) * (len(gpus) if cuda_ok else 1),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY and cuda_ok)

    val_loader = DataLoader(
        DatasetCls(config, is_train=False),
        batch_size=max(1, config.TEST.BATCH_SIZE_PER_GPU) * (len(gpus) if cuda_ok else 1),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY and cuda_ok)

    print(f'=> DataLoader batch_size={train_loader.batch_size}  workers={config.WORKERS}')

    # ------------- training loop -------------
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict, device)

        # ------- validation -------
        nme, preds = function.validate(config, val_loader, model,
                                       criterion, epoch, writer_dict, device)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        utils.save_checkpoint(
            {'state_dict': model.state_dict(),
             'epoch'     : epoch + 1,
             'best_nme'  : best_nme,
             'optimizer' : optimizer.state_dict()},
            preds, is_best, final_output_dir,
            f'checkpoint_{epoch}.pth')

    # save final
    torch.save(model.state_dict(), os.path.join(final_output_dir, 'final_state.pth'))
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
