from kornia.losses import js_div_loss_2d, SSIMLoss
import torch.nn as nn


class SelfTrainLoss(nn.Module):
    def __init__(self, loss_weights=None):
        super(SelfTrainLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.count_number = 0
        # self.l1_loss = None
        # self.l1_loss = SSIMLoss(window_size=11, reduction='mean')
        self.js_loss = js_div_loss_2d
        self.kl_loss = nn.KLDivLoss(log_target=True)
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11, reduction='mean')

        # CUD 内部平衡权重：基于实测各项原始(无系数) loss 反推，使每项加权后贡献相当(~1.0)。
        # 实测(COCO train2017, 8 样本均值，训练数据)：common≈0.038 upper≈0.038 lower≈0.037
        #                                  recon≈0.072 fusion_l1≈0.024 fusion_ssim≈0.101
        # 旧权重 20/10/10/1/100+50 使 fusion 一项占 ~90%，这里改为均衡。
        # 注意：recon 为 rec1+rec2 两个 L1 之和，故原始值约为单项的 2 倍，权重取一半。
        # fusion_ssim 波动大，权重取小，避免 fusion 项抖动。
        # 默认值；可用配置文件的 train.loss_weights 覆盖。
        _default_weights = {
            'common': 26.0,       # 0.0376 * 26 ≈ 0.98
            'upper': 26.0,        # 0.0384 * 26 ≈ 1.00
            'lower': 26.0,        # 0.0374 * 26 ≈ 0.97
            'recon': 14.0,        # 0.0718 * 14 ≈ 1.01 (recon 为两 L1 之和)
            'fusion_l1': 25.0,    # 0.0241 * 25 ≈ 0.60
            'fusion_ssim': 4.0,   # 0.1014 * 4  ≈ 0.41 (fusion 合计 ~1.01)
        }
        if loss_weights is not None:
            _default_weights.update(loss_weights)
        self.loss_weights = _default_weights

    def inital_losses(self, lossses):
        lossses['self_supervised_common_mix'] = 0
        lossses['self_supervised_upper_mix'] = 0
        lossses['self_supervised_lower_mix'] = 0
        lossses['self_supervised_recon'] = 0
        lossses['self_supervised_fusion_mix'] = 0

    def forward(self, img1, img2, gt_img, rec_img1, rec_img2, common_part, upper_part, lower_part, fusion_part):
        losses = {}
        compute_num = {}
        losses['total_loss'] = 0

        # if self.count_number < 20e3:
        #     self.count_number += 1
        #     use_unique = 0.0
        # else:
        use_unique = 1.0

        self.inital_losses(losses)
        for index in range(img1.size(0)):
            common_part_i = common_part[index].unsqueeze(0)
            upper_part_i = upper_part[index].unsqueeze(0)
            lower_part_i = lower_part[index].unsqueeze(0)
            fusion_part_i = fusion_part[index].unsqueeze(0)
            rec_img1_i = rec_img1[index].unsqueeze(0)
            rec_img2_i = rec_img2[index].unsqueeze(0)
            img1_i = img1[index].unsqueeze(0)
            img2_i = img2[index].unsqueeze(0)
            gt_i = gt_img[index].unsqueeze(0)

            mask1 = gt_i[:, 0:1, :, :]
            mask2 = gt_i[:, 1:2, :, :]
            gt_img1_i = gt_i[:, 2:5, :, :]
            gt_img2_i = gt_i[:, 5:8, :, :]
            common_mask = ((mask1 == 1.) & (mask2 == 1.)).float()
            gt_common_part = common_mask * gt_img1_i
            gt_upper_part = (mask1 - common_mask).abs() * gt_img1_i
            gt_lower_part = (mask2 - common_mask).abs() * gt_img2_i

            w = self.loss_weights
            self_supervised_common_mix_loss = w['common'] * self.l1_loss(common_part_i, gt_common_part)
            losses['self_supervised_common_mix'] += self_supervised_common_mix_loss
            self_supervised_upper_mix_loss = w['upper'] * self.l1_loss(upper_part_i, gt_upper_part) * use_unique
            losses['self_supervised_upper_mix'] += self_supervised_upper_mix_loss
            self_supervised_lower_mix_loss = w['lower'] * self.l1_loss(lower_part_i, gt_lower_part) * use_unique
            losses['self_supervised_lower_mix'] += self_supervised_lower_mix_loss
            self_supervised_recon = w['recon'] * (self.l1_loss(rec_img1_i, gt_img1_i) + self.l1_loss(rec_img2_i, gt_img2_i))
            losses['self_supervised_recon'] += self_supervised_recon

            self_supervised_fusion_mix_loss = w['fusion_l1'] * self.l1_loss(gt_img1_i, fusion_part_i) \
                + w['fusion_ssim'] * self.ssim_loss(gt_img1_i, fusion_part_i)

            losses['self_supervised_fusion_mix'] += self_supervised_fusion_mix_loss
            losses['total_loss'] += self_supervised_common_mix_loss + self_supervised_upper_mix_loss \
                                         + self_supervised_lower_mix_loss + self_supervised_fusion_mix_loss + self_supervised_recon

        return losses
