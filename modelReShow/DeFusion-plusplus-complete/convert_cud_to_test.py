"""Convert a Stage-1 UCMIMNet training checkpoint to MUCMIMNetTest format.

UCMIMNet's state_dict is a superset of MUCMIMNetTest: the training model adds
`mask_token`, `decoder_pos_embed` and `mim_decoder_*` (the MFM branch).  We
drop those keys so the checkpoint can be loaded with strict=True by test.py.

Usage:  python convert_cud_to_test.py <src.pth> <dst.pth>
"""
import sys

import torch

from models.MUCMIModelTest import MUCMIMNetTest


def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else 'experiments/MSRS_CUD_demo/models/latest.pth'
    dst_path = sys.argv[2] if len(sys.argv) > 2 else 'pretrained/msrs_cud_demo.pth'

    src = torch.load(src_path, map_location='cpu')
    sd = src['state_dict']

    drop_prefixes = ('mask_token', 'decoder_pos_embed', 'mim_decoder_')
    clean_sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}

    model = MUCMIMNetTest()
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print('dropped {} training-only keys'.format(len(sd) - len(clean_sd)))
    print('missing  :', missing)
    print('unexpected:', unexpected)

    torch.save({'state_dict': clean_sd, 'epoch': src.get('epoch', 0)}, dst_path)
    print('saved ->', dst_path)


if __name__ == '__main__':
    main()
