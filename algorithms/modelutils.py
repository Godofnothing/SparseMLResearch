import os
import sys

import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def is_depthwise(module):
    return type(module) == nn.Conv2d and module.in_channels == module.groups


@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for x, y in dataloader:
        preds.append(torch.argmax(model(x.to(dev)), 1))
        ys.append(y.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('%.2f' % acc)
    if model.training:
        model.train()

@torch.no_grad()
def test_yolo(model, dataloader):
    if 'yolov5' not in sys.path:
        sys.path.append('yolov5')

    import json
    from pathlib import Path
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from yolov5.utils.general import coco80_to_coco91_class, non_max_suppression, scale_coords, xywh2xyxy
    from yolov5.utils.metrics import ap_per_class
    from yolov5.val import process_batch, save_one_json

    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device

    conf_thres = .001
    iou_thres = .65

    iouv = torch.Tensor([.5]) 
    niou = iouv.numel()
    class_map = coco80_to_coco91_class()
    jdict = []
    names = {k: v for k, v in enumerate(model.names)}

    for i, (im, targets, paths, shapes) in enumerate(dataloader): 
        im = im.to(dev)
        targets = targets.to(dev)
        out, _ = model(im)
        nb, _, height, width = im.shape
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(dev)
        out = non_max_suppression(
            out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False
        )
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path, shape = Path(paths[si]), shapes[si][0]
            if len(pred) == 0:
                continue
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            save_one_json(predn, jdict, path, class_map)

    anno_json = dataloader.dataset.original.path.replace(
        'images/val2017', 'annotations/instances_val2017.json'
    )
    import random
    pred_json = 'yolo-preds-for-eval-%d.json' % random.randint(0, 10 ** 6)
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    eval = COCOeval(anno, pred, 'bbox')
    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.original.img_files]
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]
    print(100 * map50)
    os.remove(pred_json)

    if train:
        model.train()

@torch.no_grad()
def test_bertsquad(model, _):
    import bertsquad
    bertsquad.test(model)

def get_test(name):
    if 'yolo' in name:
        return test_yolo
    if 'bertsquad' in name:
        return test_bertsquad
    return test


def run(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if loss:
        return nn.functional.cross_entropy(out, batch[1].to(dev)).item() * batch[0].shape[0]
    return out

def run_yolo(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if not model.training:
        out = out[1]
    if loss:
        return model.computeloss(out, batch[1].to(dev))[0].item()
    return torch.cat([o.flatten() for o in out])

def run_bert(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    for k, v in batch.items():
        batch[k] = v.to(DEV)
    if retmoved:
        return batch
    out = model(**batch)
    if loss:
        return out['loss'].item() * batch[k].shape[0]
    return torch.cat([out['start_logits'], out['end_logits']])

def get_run(model):
    if 'yolo' in model:
        return run_yolo
    if 'bert' in model:
        return run_bert
    return run


def get_yolo(var):
    if 'yolov5' not in sys.path:
        sys.path.append('yolov5')
    from yolov5.models.yolo import Model
    from yolov5.utils.downloads import attempt_download
    weights = attempt_download(var + '.pt')
    ckpt = torch.load(weights, map_location=DEV)
    model = Model(ckpt['model'].yaml)
    csd = ckpt['model'].float().state_dict()
    model.load_state_dict(csd, strict=False)
    from yolov5.utils.loss import ComputeLoss
    model.hyp = {
        'box': .05, 'cls': .5, 'cls_pw': 1., 'obj': 1., 'obj_pw': 1., 'fl_gamma': 0., 'anchor_t': 4
    }
    model = model.to(DEV)
    model.computeloss = ComputeLoss(model)
    return model

def get_bertsquad(layers=12):
    import bertsquad
    return bertsquad.get_model(layers=layers)

class SplitAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def split_qkv(model):
    for block in model.blocks:
        dim = block.attn.qkv.in_features
        attention = SplitAttention(
            dim,
            num_heads=block.attn.num_heads,
            qkv_bias=hasattr(block.attn.qkv, 'bias') and block.attn.qkv.bias is not None,
            attn_drop=block.attn.attn_drop.p,
            proj_drop=block.attn.proj_drop.p
        )
        attention.q.weight.data = block.attn.qkv.weight.data[(0 * dim):(1 * dim), :]
        attention.q.bias.data = block.attn.qkv.bias.data[(0 * dim):(1 * dim)]
        attention.k.weight.data = block.attn.qkv.weight.data[(1 * dim):(2 * dim), :]
        attention.k.bias.data = block.attn.qkv.bias.data[(1 * dim):(2 * dim)]
        attention.v.weight.data = block.attn.qkv.weight.data[(2 * dim):(3 * dim), :]
        attention.v.bias.data = block.attn.qkv.bias.data[(2 * dim):(3 * dim)]
        attention.proj = block.attn.proj
        attention = attention.to(block.attn.qkv.weight.device)
        block.attn = attention
    return model

from torchvision.models import resnet18, resnet34, resnet50, resnet101 
from timm import create_model 

get_models = {
    'rn18': lambda: resnet18(pretrained=True),
    'rn34': lambda: resnet34(pretrained=True),
    'rn50': lambda: resnet50(pretrained=True),
    'rn101': lambda: resnet101(pretrained=True),
    'yolov5n': lambda: get_yolo('yolov5n'),
    'yolov5s': lambda: get_yolo('yolov5s'),
    'yolov5m': lambda: get_yolo('yolov5m'),
    'yolov5l': lambda: get_yolo('yolov5l'),
    'bertsquad': lambda: get_bertsquad(),
    'bertsquad6': lambda: get_bertsquad(6),
    'bertsquad3': lambda: get_bertsquad(3),
    'deit-tiny': lambda: split_qkv(create_model('deit_tiny_patch16_224', pretrained=True))
}

def get_model(model):
    model = get_models[model]()
    model = model.to(DEV)
    model.eval()
    return model


def get_functions(model):
    return lambda: get_model(model), get_test(model), get_run(model)


def firstlast_names(model):
    if 'rn' in model:
        return ['conv1', 'fc']
    if 'bertsquad' in model:
        return [
            'bert.embeddings.word_embeddings',
            'bert.embeddings.token_type_embeddings',
            'qa_outputs'
        ]
    if 'yolo' in model:
        lastidx = {'n': 24}[model[6]]
        return ['model.0.conv'] + ['model.%d.m.%d' % (lastidx, i) for i in range(3)]