from collections import * 

import torch
import torch.nn as nn


def get_flops(layers, model, sample, run):
    flops = {}
    def record_flops(name):
        def tmp(layer, inp, out):
            inp = inp[0]
            if isinstance(layer, nn.Conv2d):
                flops[name] = inp.shape[2] * inp.shape[3]
                flops[name] *= layer.weight.numel()
                stride = list(layer.stride)
                flops[name] //= stride[0] * stride[1] 
            if isinstance(layer, nn.Linear):
                flops[name] = layer.weight.numel()
        return tmp
    handles = []
    for name, layer in layers.items():
        if hasattr(layer, 'module'):
            layer.module.register_forward_hook(record_flops(name))
        else:
            layer.register_forward_hook(record_flops(name))
    with torch.no_grad():
        run(model, sample)
    for h in handles:
        h.remove()
    return flops

def load_errors(sds, path):
    errors = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            name = lines[i].strip()
            errors[name] = {}
            i += 1
            for _ in range(len(sds)):
                err, level = lines[i].strip().split(' ')
                errors[name][level] = float(err)
                i += 1
    return errors


class UnstrDatabase:

    def __init__(self, path, model):
        self.db = defaultdict(OrderedDict)
        denselayers = find_layers(model)
        dev = next(iter(denselayers.values())).weight.device
        for f in os.listdir(path):
            sparsity = '0.' + f.split('.')[0].split('_')[1]
            sd = torch.load(os.path.join(path, f), map_location=dev)
            for layer in denselayers:
                self.db[layer][sparsity] = sd[layer + '.weight']

    def layers(self):
        return list(self.db.keys())

    def load(self, layers, name, config='', sd=None):
        if sd is not None:
            layers[name].weight.data = sd[name + '.weight']
            return
        layers[name].weight.data = self.db[name][config]

    def stitch(self, layers, config):
        for name, layer in layers.items():
            self.load(layers, name, config[name])

    def load_errors(self, path):
        return load_errors(self.sds, path)

    def get_params(self, layers):
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = torch.sum(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res

    def get_flops(self, layers, model, sample, run):
        flops = get_flops(layers, model, sample, run)
        res = {}
        for name in layers:
            res[name] = {}
            for sparsity in self.sds:
                res[name][sparsity] = flops[name] * torch.mean(
                    (self.sds[sparsity][name + '.weight'] != 0).float()
                ).item()
        return res


def compute_squared(db, get_model, dataloader, run, filename, dataloader_passes=1):
    modeld = get_model()
    modelp = get_model()
    layersd = find_layers(modeld)
    layersp = find_layers(modelp)

    errs = {n: {} for n in db.layers()}
    def accumerrs(name):
        def tmp(layer, inp, out):
            errs[name]['dense'] = errs[name].get('dense', 0) + torch.sum(out.data ** 2).item()
            for config in sorted(db.db[name]):
                db.load(layersp, name, config)
                errs[name][config] = errs[name].get(config, 0) + torch.sum((layersp[name](inp[0].data) - out.data) ** 2).item()
        return tmp
    for name in db.layers():
        layersd[name].register_forward_hook(accumerrs(name))

    with torch.no_grad():
        for i in range(dataloader_passes):
            for j, batch in enumerate(dataloader):
                print(i, j)
                run(modeld, batch)

    with open(filename, 'w') as f:
        for name in errs:
            f.write(name + '\n')
            for config in sorted(errs[name]):
                if config != 'dense':
                    f.write('%.6f %s\n' % (errs[name][config] / errs[name]['dense'], config))

def compute_loss(db, model, dataloader, run, nsamples, filename):
    layers = find_layers(model)
    sd = model.state_dict()
    errs = {n: {} for n in db.layers()}
    baseloss = 0

    for i, batch in enumerate(dataloader):
        print(i)
        with torch.no_grad():
            baseloss += run(model, batch, loss=True)
            for name in db.layers():
                print(name)
                for config in sorted(db.db[name]):
                    db.load(layers, name, config)
                    errs[name][config] = errs[name].get(config, 0) + run(model, batch, loss=True)
                db.load(layers, name, sd=sd)
    baseloss /= nsamples
    for name in errs:
        for config in errs[name]:
            errs[name][config] /= nsamples

    with open(filename, 'w') as f:
        for name in errs:
            f.write(name + '\n')
            for config in sorted(errs[name]):
                f.write('%+.6f %s\n' % (errs[name][config] - baseloss, config))


if __name__ == '__main__':
    import argparse

    from datautils import *
    from modelutils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('database', type=str)
    parser.add_argument('mode', choices=['squared', 'loss'])
    parser.add_argument('savescores', type=str)

    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=1024)

    args = parser.parse_args()

    get_model, test, run = get_functions(args.model)
    dataloader, testloader = get_loaders(
        args.dataset, path=args.datapath,
        nsamples=args.nsamples, seed=args.seed,
        noaug=True
    )

    model = get_model()
    db = UnstrDatabase(args.database, model)

    if args.mode == 'squared':
        compute_squared(db, get_model, dataloader, run, args.savescores)
    if args.mode == 'loss':
        compute_loss(db, model, dataloader, run, args.nsamples, args.savescores)
