import os
import sys
import numpy as np
import torch
import argparse
import logging
from PIL import Image
from model import ZSXENetFinetune
from multi_read_data import DataLoader
from thop import profile


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_model_flops(model, input_tensor):
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops / 1e9


def main():
    parser = argparse.ArgumentParser("ZS-XENet")
    parser.add_argument('--data_path_test_low', type=str, default='image')
    parser.add_argument('--save', type=str, default='./results/')
    parser.add_argument('--model_test', type=str, default='weights/zsxenet_best.pt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d %I:%M:%S %p')

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)

    # ===== Data =====
    TestDataset = DataLoader(img_dir=args.data_path_test_low, task='test')
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False
    )

    # ===== Model =====
    model = ZSXENetFinetune(args.model_test)
    model = model.cuda().eval()

    total_params = calculate_model_parameters(model)
    logging.info(f"Total parameters: {total_params / 1e6:.4f} M")

    # ===== Testing =====
    with torch.no_grad():
        base_dir = os.path.join(args.save, 'test')
        enhance_dir = os.path.join(base_dir, 'enhance')
        denoise_dir = os.path.join(base_dir, 'denoise')
        os.makedirs(enhance_dir, exist_ok=True)
        os.makedirs(denoise_dir, exist_ok=True)

        for _, (input, img_name) in enumerate(test_queue):
            input = input.cuda(non_blocking=True)
            enhance, output = model(input)
            name = os.path.splitext(os.path.basename(img_name[0]))[0]

            enhance_img = save_images(enhance)
            output_img = save_images(output)

            # Ir (enhance) and Ir_refined (denoise)
            Image.fromarray(output_img).save(os.path.join(denoise_dir, f'{name}_denoise.png'))
            Image.fromarray(enhance_img).save(os.path.join(enhance_dir, f'{name}_enhance.png'))

    logging.info("Testing completed.")


if __name__ == '__main__':
    main()
