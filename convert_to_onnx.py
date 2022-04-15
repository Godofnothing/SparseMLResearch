import os
import timm
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from engine import val_epoch


def parse_args():
    parser = argparse.ArgumentParser('Low-rank + Sparse weight approximation.', add_help=False)
    # Model
    parser.add_argument('--model', default='deit_small_patch16_224', type=str)
    # Path to data
    parser.add_argument('--data-dir', required=True, type=str)
    # Path to checkpoint
    parser.add_argument('--checkpoint-path', required=True, type=str)
    # ONNX export params
    parser.add_argument('--img-size', default=224, type=int)
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    # Evaluator parameters
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('-vb', '--val_batch_size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    # Save arguments
    parser.add_argument('--onnx-dir', default='./onnx_models', type=str, 
                        help='dir to save results')
    parser.add_argument('--onnx-suffix', default='', type=str, 
                        help='additional suffix for saved model')

    
    args = parser.parse_args()
    return args


def convert_to_onnx(model, export_name: str, input_size=(1, 3, 224, 224), device='cpu'):
    # put model to the evaluation mode
    model.eval()
    # create dummy input
    dummy_input = torch.randn(input_size, device=device)
    # define input and output names
    input_names  = ["model_input"]
    output_names = ["model_output"]
    # export the model
    torch.onnx.export(
        model, 
        dummy_input, 
        export_name, 
        verbose=True, 
        input_names=input_names, 
        output_names=output_names
    )

if __name__ == '__main__':
    # parse args
    args = parse_args()
    # choose device (needed only if evaluation selected)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # validation transforms
    val_transforms =  T.Compose([
        T.Resize(int(8 * args.img_size / 7)),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # validation dataset
    val_dataset = ImageFolder(
        root=f'{args.data_dir}/val', transform=val_transforms
    )
    # validation loader
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers)
    # loss function
    criterion = nn.CrossEntropyLoss()
    # get model
    model = timm.create_model(args.model)
    # load checkpoint
    state_dict = torch.load(f'{args.checkpoint_path}', map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    # model to device
    model = model.to(device)
    if args.evaluate:
        val_acc = val_epoch(model, val_loader, criterion, device=device)['acc']
        print(f'Test accuracy is {val_acc:.3f}')
    # create save dir if neeeded
    os.makedirs(f'{args.onnx_dir}', exist_ok=True)
    # convert to onnx and save
    convert_to_onnx(
        model, 
        export_name=f'{args.onnx_dir}/{args.model}{args.onnx_suffix}.onnx',
        input_size=(args.batch_size, 3, args.img_size, args.img_size),
        device=device
    )
        