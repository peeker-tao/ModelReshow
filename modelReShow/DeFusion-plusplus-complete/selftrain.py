import argparse
import os

import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import build_code_arch
from data.self_mixpretrain_dataset import TrainDataset
from models.UCMIModel import UCMIMNet
from loss.mix_fp_loss import SelfTrainLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option yaml file.')
    train_args = parser.parse_args()

    opt, resume_state = build_code_arch.build_resume_state(train_args)
    opt, logger, tb_logger = build_code_arch.build_logger(opt)

    train_dataset = TrainDataset(opt['dataset']['train'])
    train_loader = DataLoader(
        train_dataset, batch_size=opt['dataset']['train']['batch_size'], shuffle=True,
        num_workers=opt['dataset']['train']['workers'], pin_memory=True)
    logger.info('Number of train images: {:,d}'.format(len(train_dataset)))

    model = UCMIMNet()

    optimizer = AdamW(model.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']),
                      lr=opt['train']['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                         milestones=opt['train']['lr_steps'],
                                         gamma=opt['train']['lr_gamma'])
    writer = SummaryWriter()
    model = model.cuda()
    model.train()

    if resume_state:
        logger.info('Resuming training from epoch: {}.'.format(resume_state['epoch']))
        start_epoch = resume_state['epoch'] + 1
        optimizer.load_state_dict(resume_state['optimizers'])
        model.load_state_dict(resume_state['state_dict'])
    else:
        start_epoch = 0

    criterion = SelfTrainLoss()
    max_steps = len(train_loader)

    logger.info('Start training from epoch: {:d}'.format(start_epoch))
    logger.info('# network parameters: {}'.format(
        sum(param.numel() for param in model.parameters())))
    total_epochs = opt['train']['epoch']

    for epoch in range(start_epoch, total_epochs + 1):
        criterion.is_train = True
        for index, train_data in tqdm(enumerate(train_loader)):
            o_img, v_img, gt_img, train_type = train_data
            o_img = o_img.cuda()
            v_img = v_img.cuda()
            gt_img = gt_img.cuda()

            rec_img1, rec_img2, common_part, upper_part, lower_part, fusion_part = model(o_img, v_img)

            losses = criterion(img1=o_img, img2=v_img, gt_img=gt_img,
                               rec_img1=rec_img1, rec_img2=rec_img2,
                               common_part=common_part, upper_part=upper_part,
                               lower_part=lower_part, fusion_part=fusion_part)
            grad_loss = losses["total_loss"]
            optimizer.zero_grad()
            grad_loss.backward()
            optimizer.step()
            current_step = epoch * max_steps + index

            message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                epoch, current_step, scheduler.get_last_lr()[0])
            for k, v in losses.items():
                v = v.cpu().item()
                message += '{:s}: {:.4e} '.format(k, v)
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar(k, v, current_step)
            logger.info(message)

        scheduler.step()

        if epoch % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info('Saving models and training states.')
            save_filename = '{}_{}.pth'.format(epoch, 'models')
            save_path = os.path.join(opt['path']['models'], save_filename)
            save_checkpoint = {'state_dict': model.state_dict(),
                               'optimizers': optimizer.state_dict(),
                               'schedulers': scheduler.state_dict(),
                               'epoch': epoch}
            torch.save(save_checkpoint, save_path)
            torch.cuda.empty_cache()

    logger.info('Saving the final model.')
    save_filename = 'latest.pth'
    save_path = os.path.join(opt['path']['models'], save_filename)
    save_checkpoint = {"state_dict": model.state_dict(),
                       'optimizers': optimizer.state_dict(),
                       'schedulers': scheduler.state_dict(),
                       "epoch": opt['train']['epoch']}
    torch.save(save_checkpoint, save_path)
    logger.info('End of training.')
    tb_logger.close()


if __name__ == '__main__':
    main()
