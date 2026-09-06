"""DeFusion++ Stage-2 (MCUD) multi-modality training.

Trains ``MUCMIMNet`` with two self-supervised objectives:
    * CUD   (single-modality, grid-masked self-supervision)  -> ``SelfTrainLoss``
    * MCUD  (paired IR/VI latent alignment)                  -> ``MultiModalLoss``

Run:  python selftrain_multimodal.py -opt option/train/COCO_MSRS_MCUD_demo.yaml
"""
import argparse
import logging
import math
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

from option import options as option
from utils import build_code_arch, util
from utils.util import MetricLogger, SmoothedValue
from data.self_mixpretrain_dataset import TrainDataset as NoiseTrainDataset
from data.irvis_train_dataset import IRVISTrainDataset
from data.irvis_val_dataset import IRVIValDataset
from models.MMUCMIModel import MUCMIMNet
from loss.mix_fp_loss import SelfTrainLoss
from loss.multimodalloss import MultiModalLoss


def adjust_learning_rate(optimizer, base_lr, step, warmup_steps, min_lr, total_steps):
    """MuM 风格：warmup 之后 half-cycle cosine 衰减。"""
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def build_noise_opt(dataset_opt):
    """把 noise_* key 映射为 NoiseTrainDataset 期望的 dataroot/trainpairs。"""
    noise_opt = dict(dataset_opt)
    noise_opt['dataroot'] = dataset_opt.get('noise_dataroot', dataset_opt.get('dataroot'))
    noise_opt['trainpairs'] = dataset_opt.get('noise_trainpairs', dataset_opt.get('trainpairs'))
    return noise_opt


def evaluate_val(model, cud_criterion, mcud_criterion, cud_loader, irvis_loader, val_batches,
                 cud_weight=1.0, mcud_weight=8.0, mode='mcud'):
    """在验证集（MSRS test）上计算加权平均 total loss。
    mode='cud' 时只评估 CUD 分支（纯 CUD 阶段）。"""
    model.eval()
    total = 0.0
    count = 0
    cud_iter = iter(cud_loader)
    irvis_iter = iter(irvis_loader) if mode == 'mcud' else None
    with torch.no_grad():
        for _ in range(val_batches):
            try:
                o_img, v_img, gt_img, _ = next(cud_iter)
                if mode == 'mcud':
                    ir_img, vis_img = next(irvis_iter)
            except StopIteration:
                break
            o_img, v_img, gt_img = o_img.cuda(), v_img.cuda(), gt_img.cuda()

            # ---- CUD branch ----
            rec_img1, rec_img2, common_part, upper_part, lower_part, fusion_part = model(o_img, v_img)
            cud_losses = cud_criterion(img1=o_img, img2=v_img, gt_img=gt_img,
                                       rec_img1=rec_img1, rec_img2=rec_img2,
                                       common_part=common_part, upper_part=upper_part,
                                       lower_part=lower_part, fusion_part=fusion_part)
            loss_this = cud_weight * cud_losses['total_loss']

            # ---- MCUD branch ----
            if mode == 'mcud':
                ir_img, vis_img = ir_img.cuda(), vis_img.cuda()
                modality_gt1, modality_gt2, modality_predict1, modality_predict2, com_img_w1, com_img_w2 = \
                    model(ir_img, vis_img, modality='irvis')
                mcud_losses = mcud_criterion(modality_gt1, modality_gt2, modality_predict1,
                                             modality_predict2, com_img_w1, com_img_w2)
                loss_this = loss_this + mcud_weight * mcud_losses['total_loss']

            total += loss_this.item()
            count += 1
    model.train()
    return total / count if count else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option yaml file.')
    train_args = parser.parse_args()

    opt, resume_state = build_code_arch.build_resume_state(train_args)

    # MuM 风格日志（时间戳 - 级别: 消息）
    util.setup_logger('defusion-plusplus', opt['path']['log'], 'train_' + opt['name'],
                      level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('defusion-plusplus')
    logger.info(option.dict2str(opt))
    util.set_random_seed(opt['train']['manual_seed'])
    torch.backends.cudnn.benchmark = True

    train_opt = opt['dataset']['train']
    val_opt = opt['dataset']['val']
    train_cfg = opt['train']

    # ---- 训练模式 ----
    # mode='cud'  (阶段1)：只跑 CUD（单模态 grid-mask 分解+融合），不创建教师，省显存
    # mode='mcud' (阶段2)：CUD + MCUD 联合训练（需要冻结 MAE 教师）
    train_mode = train_cfg.get('mode', 'mcud')
    assert train_mode in ('cud', 'mcud'), "train.mode must be 'cud' or 'mcud', got: {}".format(train_mode)
    if train_mode == 'cud':
        use_teacher = False
        logger.info('Training mode: CUD-ONLY (阶段1, 无教师, 省显存)')
    else:
        use_teacher = True
        if not (train_opt.get('teacher1_pth') and train_opt.get('teacher2_pth')):
            logger.warning('MCUD mode requires teacher1_pth/teacher2_pth in dataset.train; '
                           'teachers will be random-initialized (frozen) if missing!')
        logger.info('Training mode: CUD + MCUD (阶段2, 含冻结 MAE 教师)')

    # ---- loss 平衡权重 ----
    # CUD 内部已在 SelfTrainLoss 内按实测(COCO)量级均衡，均衡后 CUD total ≈ 5.0/样本；
    # MCUD 原始 total ≈ 0.33/样本，故 CUD:MCUD ≈ 15:1。
    # 乘 mcud_weight=8 后 CUD:MCUD ≈ 1.9:1（CUD 多一点）。
    cud_weight = train_cfg.get('cud_loss_weight', 1.0)
    mcud_weight = train_cfg.get('mcud_loss_weight', 8.0)
    loss_weights = train_cfg.get('loss_weights', None)
    logger.info('Loss weights: cud_weight={}, mcud_weight={}, loss_weights={}'.format(
        cud_weight, mcud_weight, loss_weights))

    # ---- CUD 分支数据（单模态 grid-mask）----
    noise_dataset = NoiseTrainDataset(build_noise_opt(train_opt))
    noise_loader = DataLoader(noise_dataset, batch_size=train_opt['batch_size'],
                              shuffle=True, num_workers=train_opt['workers'], pin_memory=True)
    logger.info('Number of CUD train images: {:,d}'.format(len(noise_dataset)))

    # ---- MCUD 分支数据（成对 IR/VI，仅 mcud 模式）----
    irvis_loader = None
    if train_mode == 'mcud':
        irvis_dataset = IRVISTrainDataset(train_opt)
        irvis_loader = DataLoader(irvis_dataset, batch_size=train_opt['batch_size'],
                                  shuffle=True, num_workers=train_opt['workers'], pin_memory=True)
        logger.info('Number of MCUD train pairs: {:,d}'.format(len(irvis_dataset)))

    model = MUCMIMNet(teacher1_pth=train_opt.get('teacher1_pth') if use_teacher else None,
                      teacher2_pth=train_opt.get('teacher2_pth') if use_teacher else None,
                      use_teacher=use_teacher)
    optimizer = AdamW(model.parameters(), betas=(train_cfg['beta1'], train_cfg['beta2']),
                      lr=train_cfg['lr'])
    model = model.cuda()
    model.train()

    start_step = 0
    if resume_state:
        start_step = resume_state.get('step', resume_state.get('epoch', 0))
        optimizer.load_state_dict(resume_state['optimizers'])
        model.load_state_dict(resume_state['state_dict'])
        logger.info('Resuming training from step: {}.'.format(start_step))
    else:
        # path.init_from：从上一个阶段 checkpoint 冷启动（只加载网络权重，重置 optimizer/step）
        init_from = opt['path'].get('init_from', None)
        if init_from and os.path.exists(init_from):
            ckpt = torch.load(init_from, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            logger.info('Initialized from {}: loaded {} keys, missing {} keys (MCUD-only/teacher, '
                        'kept random/pretrained), unexpected {} keys'.format(
                            init_from, len(sd) - len(missing), len(missing), len(unexpected)))
            if missing:
                logger.info('Missing keys (e.g. teacher1/2.*): first 5 -> {}'.format(list(missing)[:5]))
            start_step = 0
        elif init_from:
            logger.warning('init_from path not found, training from scratch: {}'.format(init_from))

    cud_criterion = SelfTrainLoss(loss_weights=loss_weights)
    mcud_criterion = MultiModalLoss()

    # ---- 验证集 dataloaders（MSRS test）----
    val_cud_dataset = NoiseTrainDataset(build_noise_opt(val_opt))
    val_cud_loader = DataLoader(val_cud_dataset, batch_size=val_opt['batch_size'],
                                shuffle=False, num_workers=val_opt['workers'], pin_memory=True)
    val_irvis_loader = None
    if train_mode == 'mcud':
        val_irvis_dataset = IRVIValDataset(val_opt)
        val_irvis_loader = DataLoader(val_irvis_dataset, batch_size=val_opt['batch_size'],
                                      shuffle=False, num_workers=val_opt['workers'], pin_memory=True)

    # ---- wandb ----
    wandb_mode = 'online' if opt.get('track_wandb', True) else 'disabled'
    wandb.init(
        name=opt['name'],
        project=opt.get('wandb_project', 'DeFusion-plusplus'),
        entity=opt.get('wandb_entity') or None,
        config=dict(opt),
        mode=wandb_mode,
    )

    total_steps = train_cfg['steps']
    warmup_steps = train_cfg['warmup_steps']
    checkpoint_steps = train_cfg['checkpoint_steps']
    log_steps = train_cfg['log_steps']
    base_lr = train_cfg['lr']
    min_lr = train_cfg.get('min_lr', 0.0)
    max_grad_norm = train_cfg.get('max_grad_norm', 20)
    val_batches = val_opt.get('val_batches', 50)
    model_prefix = 'Defusion-plusplus'

    logger.info('# network parameters: {}'.format(
        sum(param.numel() for param in model.parameters())))
    logger.info('Start training: {} steps, warmup {} steps, log every {} steps, '
                'checkpoint every {} steps'.format(total_steps, warmup_steps, log_steps, checkpoint_steps))

    noise_iter = iter(noise_loader)
    irvis_iter = iter(irvis_loader) if train_mode == 'mcud' else None
    start_time = time.time()

    # ---- MuM 风格训练日志（MetricLogger）----
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Training'
    iter_time = SmoothedValue(fmt='{avg:.6f}')
    data_time = SmoothedValue(fmt='{avg:.6f}')

    step = start_step
    end = time.time()
    while step <= total_steps:
        data_time.update(time.time() - end)
        try:
            o_img, v_img, gt_img, _ = next(noise_iter)
        except StopIteration:
            noise_iter = iter(noise_loader)
            o_img, v_img, gt_img, _ = next(noise_iter)
        if train_mode == 'mcud':
            try:
                ir_img, vis_img = next(irvis_iter)
            except StopIteration:
                irvis_iter = iter(irvis_loader)
                ir_img, vis_img = next(irvis_iter)

        o_img, v_img, gt_img = o_img.cuda(), v_img.cuda(), gt_img.cuda()
        current_batch_size = o_img.shape[0]
        current_lr = adjust_learning_rate(optimizer, base_lr, step, warmup_steps, min_lr, total_steps)

        # ---- CUD branch ----
        rec_img1, rec_img2, common_part, upper_part, lower_part, fusion_part = model(o_img, v_img)
        cud_losses = cud_criterion(img1=o_img, img2=v_img, gt_img=gt_img,
                                   rec_img1=rec_img1, rec_img2=rec_img2,
                                   common_part=common_part, upper_part=upper_part,
                                   lower_part=lower_part, fusion_part=fusion_part)
        total_loss = cud_weight * cud_losses['total_loss']

        # ---- MCUD branch（仅 mcud 模式）----
        mcud_losses = None
        if train_mode == 'mcud':
            ir_img, vis_img = ir_img.cuda(), vis_img.cuda()
            modality_gt1, modality_gt2, modality_predict1, modality_predict2, com_img_w1, com_img_w2 = \
                model(ir_img, vis_img, modality='irvis')
            mcud_losses = mcud_criterion(modality_gt1, modality_gt2, modality_predict1,
                                         modality_predict2, com_img_w1, com_img_w2)
            total_loss = total_loss + mcud_weight * mcud_losses['total_loss']
        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            logger.error('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        torch.cuda.synchronize()
        iter_time.update(time.time() - end)

        metric_logger.update(loss=loss_value)
        metric_logger.update(grad=grad_norm)
        metric_logger.update(batch_size=current_batch_size)
        metric_logger.update(lr=current_lr)

        if step % log_steps == 0:
            logger.info(metric_logger.log_message(step, total_steps, header, iter_time, data_time))
            log_dict = {
                'loss': loss_value,
                'lr': current_lr,
                'grad': grad_norm,
                'cud_loss': cud_losses['total_loss'].item(),
            }
            if train_mode == 'mcud':
                log_dict['mcud_loss'] = mcud_losses['total_loss'].item()
            wandb.log(log_dict)

        if step % checkpoint_steps == 0 and step != 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            val_loss = evaluate_val(model, cud_criterion, mcud_criterion,
                                    val_cud_loader, val_irvis_loader, val_batches,
                                    cud_weight, mcud_weight, mode=train_mode)
            print('[step {}] val loss: {:.4f}'.format(step, val_loss))
            logger.info('[step {}] val loss: {:.4f}'.format(step, val_loss))
            wandb.log({'val_loss': val_loss, 'val_step': step})

            ckpt_name = 'checkpoint-{}-{}-step{}-loss{:.4f}-valloss{:.4f}.pth'.format(
                model_prefix, timestamp, step, loss_value, val_loss)
            save_path = os.path.join(opt['path']['models'], ckpt_name)
            torch.save({'state_dict': model.state_dict(),
                        'optimizers': optimizer.state_dict(),
                        'step': step}, save_path)
            # 固定名 latest 便于 resume
            torch.save({'state_dict': model.state_dict(),
                        'optimizers': optimizer.state_dict(),
                        'step': step},
                       os.path.join(opt['path']['models'], 'checkpoint-last.pth'))
            logger.info('Saved checkpoint: {}'.format(save_path))

        step += 1
        end = time.time()

    # 训练结束保存最终 checkpoint（step 为最后完成的步）
    final_path = os.path.join(opt['path']['models'], 'checkpoint-last.pth')
    torch.save({'state_dict': model.state_dict(),
                'optimizers': optimizer.state_dict(),
                'step': step - 1}, final_path)
    logger.info('Saved final checkpoint: {}'.format(final_path))
    logger.info('End of training. Total time {:.1f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
