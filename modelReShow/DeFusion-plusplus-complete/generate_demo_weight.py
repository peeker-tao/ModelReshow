"""Generate a random (untrained) demo checkpoint for quick inference testing.

Run:  python generate_demo_weight.py
"""
import os
import torch

from models.MUCMIModelTest import MUCMIMNetTest


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, 'pretrained')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'demo_random.pth')

    model = MUCMIMNetTest()
    torch.save({'state_dict': model.state_dict(), 'epoch': 0}, out_path)
    print('Saved demo random checkpoint to: {}'.format(out_path))
    print('# parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))


if __name__ == '__main__':
    main()
