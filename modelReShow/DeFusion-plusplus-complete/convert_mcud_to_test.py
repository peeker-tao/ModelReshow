"""Convert a Stage-2 MCUD (MUCMIMNet) training checkpoint to MUCMIMNetTest format.

Stage-2 ``MUCMIMNet`` is a superset of the inference model ``MUCMIMNetTest``:
it additionally carries the two frozen MAE teachers (``model_encoder1/2``),
the latent predictors (``latent_predict1/2``), the unique latent norms
(``mm_unique_norm1/2``), and the masked-feature-modeling branch
(``mask_token`` / ``decoder_pos_embed`` / ``mim_decoder_*``).  Those keys are
not present in ``MUCMIMNetTest``, so they must be dropped for a strict load.

Usage:
    python convert_mcud_to_test.py <src.pth> <dst.pth>
"""
import sys

import torch

from models.MUCMIModelTest import MUCMIMNetTest


def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else \
        'experiments/COCO_MSRS_MCUD_demo/models/checkpoint-last.pth'
    dst_path = sys.argv[2] if len(sys.argv) > 2 else \
        'pretrained/msrs_mcud_trained.pth'

    src = torch.load(src_path, map_location='cpu')
    sd = src['state_dict']

    # keep only the keys the inference model actually has (robust whitelist)
    keep = set(MUCMIMNetTest().state_dict().keys())
    clean_sd = {k: v for k, v in sd.items() if k in keep}

    model = MUCMIMNetTest()
    missing, unexpected = model.load_state_dict(clean_sd, strict=True)
    print('total training keys   :', len(sd))
    print('kept inference keys   :', len(clean_sd))
    print('dropped training-only :', len(sd) - len(clean_sd))
    print('missing  :', missing)
    print('unexpected:', unexpected)

    out = {'state_dict': clean_sd, 'epoch': src.get('epoch', src.get('step', 0))}
    torch.save(out, dst_path)
    print('saved ->', dst_path)


if __name__ == '__main__':
    main()
