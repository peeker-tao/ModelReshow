import argparse
import os
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from utils import util
from data.multi_exposure_dataset import TestDataset as MEFTestDataset
from data.multi_focus_dataset import TestDataset as MFFTestDataset
from data.visir_fusion_dataset import TestDataset as IVFTestDataset
from models.MUCMIModelTest import MUCMIMNetTest
import option.options as option


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Multi Data Fusion: Path to option yaml file.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    util.mkdir_and_rename(opt['path']['results_root'])  # rename results folder if exists
    # only create real string paths (skip bool flags such as strict_load)
    util.mkdirs((path for key, path in opt['path'].items()
                 if not key == 'results_root'
                 and 'pretrain' not in key and 'resume' not in key
                 and isinstance(path, str)))
    util.setup_logger('defusion-plusplus', opt['path']['log'], 'test_' + opt['name'],
                      level=logging.INFO, screen=True, tofile=True)

    logger = logging.getLogger('defusion-plusplus')
    logger.info(option.dict2str(opt))

    torch.backends.cudnn.deterministic = True
    opt = option.dict_to_nonedict(opt)

    if 'MultiFocusFusion' in opt['name']:
        TestDataset = MFFTestDataset
    elif 'MultiExposureFusion' in opt['name']:
        TestDataset = MEFTestDataset
    elif 'VisibleInfrareFusion' in opt['name']:
        TestDataset = IVFTestDataset
    elif 'MEF' in opt['name']:
        TestDataset = MEFTestDataset
    else:
        raise ValueError('Unknown task type in name: {}'.format(opt['name']))

    dataset_opt = opt['dataset']['test']
    test_dataset = TestDataset(dataset_opt)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=dataset_opt['workers'], pin_memory=True)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_dataset)))

    model = MUCMIMNetTest()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_path = opt['path']['resume_state']
    assert resume_path and os.path.exists(resume_path), \
        'resume_state checkpoint not found: {}'.format(resume_path)
    resume_state = torch.load(resume_path, map_location=device)
    logger.info('Resuming state from epoch: {}.'.format(resume_state.get('epoch', 'unknown')))
    model.load_state_dict(resume_state['state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()

    idx = 0
    for test_data in tqdm(test_loader):
        with torch.no_grad():
            o_img, u_img, root_name = test_data

            padding_number = 16
            o_img = F.pad(o_img, (padding_number, padding_number, padding_number, padding_number), mode='reflect')
            u_img = F.pad(u_img, (padding_number, padding_number, padding_number, padding_number), mode='reflect')
            o_img = o_img.to(device)
            u_img = u_img.to(device)

            common_part, upper_part, lower_part, fusion_part = model(o_img, u_img)

            o_img = o_img[:, :, padding_number:-padding_number, padding_number:-padding_number]
            u_img = u_img[:, :, padding_number:-padding_number, padding_number:-padding_number]
            common_part = common_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
            upper_part = upper_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
            lower_part = lower_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
            fusion_part = fusion_part[:, :, padding_number:-padding_number, padding_number:-padding_number]
            logger.info('ou img {} {} {} {}'.format(o_img.shape, u_img.shape, fusion_part.shape, root_name))

            img_dir = opt['path']['test_images']

            common_img = ToPILImage()(common_part.clamp(0, 1)[0].cpu())
            common_img.save(os.path.join(img_dir, "{:s}_common.png".format(root_name[0])))

            upper_img = ToPILImage()(upper_part.clamp(0, 1)[0].cpu())
            upper_img.save(os.path.join(img_dir, "{:s}_upper.png".format(root_name[0])))

            lower_img = ToPILImage()(lower_part.clamp(0, 1)[0].cpu())
            lower_img.save(os.path.join(img_dir, "{:s}_lower.png".format(root_name[0])))

            over_img = ToPILImage()(o_img[0].cpu())
            over_img.save(os.path.join(img_dir, "{:s}_over.png".format(root_name[0])))

            under_img = ToPILImage()(u_img[0].cpu())
            under_img.save(os.path.join(img_dir, "{:s}_under.png".format(root_name[0])))

            recover = fusion_part
            recover_img = ToPILImage()(recover.clamp(0, 1)[0].cpu())
            recover_img.save(os.path.join(img_dir, "{:s}_recover.png".format(root_name[0])))

            idx += 1

    logger.info('End of testing, processed {:d} images.'.format(idx))


if __name__ == '__main__':
    main()
