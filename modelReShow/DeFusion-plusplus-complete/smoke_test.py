"""CPU 冒烟测试：验证 CUD+MCUD 双分支前向与 loss 计算（不依赖数据、不占 GPU）。"""
import torch

from models.MMUCMIModel import MUCMIMNet
from loss.mix_fp_loss import SelfTrainLoss
from loss.multimodalloss import MultiModalLoss

torch.manual_seed(0)

model = MUCMIMNet(teacher1_pth=None, teacher2_pth=None)
model.eval()
cud_criterion = SelfTrainLoss()
mcud_criterion = MultiModalLoss()

B = 2
with torch.no_grad():
    # ---- CUD 分支 ----
    o_img = torch.rand(B, 3, 224, 224)
    v_img = torch.rand(B, 3, 224, 224)
    gt_img = torch.rand(B, 8, 224, 224)
    rec_img1, rec_img2, common_part, upper_part, lower_part, fusion_part = model(o_img, v_img)
    cud_losses = cud_criterion(img1=o_img, img2=v_img, gt_img=gt_img,
                               rec_img1=rec_img1, rec_img2=rec_img2,
                               common_part=common_part, upper_part=upper_part,
                               lower_part=lower_part, fusion_part=fusion_part)
    print('CUD total loss: {:.4f}'.format(cud_losses['total_loss'].item()))

    # ---- MCUD 分支 ----
    ir_img = torch.rand(B, 3, 224, 224)
    vis_img = torch.rand(B, 3, 224, 224)
    m_gt1, m_gt2, m_p1, m_p2, cw1, cw2 = model(ir_img, vis_img, modality='irvis')
    mcud_losses = mcud_criterion(m_gt1, m_gt2, m_p1, m_p2, cw1, cw2)
    print('MCUD total loss: {:.4f}'.format(mcud_losses['total_loss'].item()))

total = cud_losses['total_loss'] + mcud_losses['total_loss']
print('CUD+MCUD total: {:.4f}'.format(total.item()))
print('冒烟测试通过 ✔')
