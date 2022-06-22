import copy
import math
import time

import torch
import torch.nn as nn


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEBUG = False 


class TrueOBS:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Accumulate in double precision
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.double)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H, percentdamp=.01):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            diagmean = torch.mean(torch.diag(H))
            print('Hessian not full rank.')
            tmp = (percentdamp * diagmean) * torch.eye(self.columns, device=self.dev)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.float()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
        return W, H, Hinv, Losses

    def prepare_iter(self, i1, parallel, W, Hinv1):
        i2 = min(i1 + parallel, self.rows)
        count = i2 - i1
        w = W[i1:i2, :]
        Hinv = Hinv1.unsqueeze(0).repeat((count, 1, 1))
        mask = torch.zeros_like(w).bool()
        rangecount = torch.arange(count, device=self.dev)
        idxcount = rangecount + i1
        return i2, count, w, Hinv, mask, rangecount, idxcount

    def prepare_sparse(self, w, mask, Hinv, H): 
        start = int(torch.min(torch.sum((w == 0).float(), 1)).item()) + 1
        for i in range(w.shape[0]):
            tmp = w[i] == 0
            H1 = H.clone()
            H1[tmp, :] = 0
            H1[:, tmp] = 0
            H1[tmp, tmp] = 1
            Hinv[i] = self.invert(H1)
            mask[i, torch.nonzero(tmp, as_tuple=True)[0][:(start - 1)]] = True
        return start

    def prepare_unstr(self, parallel=32):
        W, H, Hinv1, Losses = self.prepare()

        self.Losses = Losses
        self.Traces = []

        for i1 in range(0, self.rows, parallel):
            i2, count, w, Hinv, mask, rangecount, idxcount = self.prepare_iter(i1, parallel, W, Hinv1)
            start = self.prepare_sparse(w, mask, Hinv, H) 

            Trace = torch.zeros((self.columns + 1, count, self.columns), device=self.dev)
            Trace[0, :, :] = w
            Trace[:start, :, :] = w

            tick = time.time()

            for zeros in range(start, self.columns + 1):
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = (w ** 2) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores, 1)
                self.Losses[i1:i2, zeros] = scores[rangecount, j]
                row = Hinv[rangecount, j, :]
                d = diag[rangecount, j]
                w -= row * (w[rangecount, j] / d).unsqueeze(1)
                mask[rangecount, j] = True
                w[mask] = 0
                Trace[zeros, :, :] = w
                if zeros == self.columns:
                    break
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))
            self.Losses[i1:i2, :] /= 2
            self.Traces.append(Trace.cpu())

            torch.cuda.synchronize()
            print('%04d %04d time %.2f' % (i1, i2, time.time() - tick))

    def prune_unstr(self, sparsities):
        parallel = self.Traces[0].shape[1]
        blockcount = self.Traces[0].shape[0] - 1
        losses = self.Losses[:, 1:].reshape(-1)
        order = torch.argsort(losses)
        Ws = [torch.zeros((self.rows, self.columns), device=self.dev) for _ in sparsities]
        losses = [0] * len(sparsities)
        for i in range(self.rows):
            if i % parallel == 0:
                Trace = self.Traces[i // parallel].to(self.dev)
            for j, sparsity in enumerate(sparsities):
                count = int(math.ceil(self.rows * blockcount * sparsity))
                perrow = torch.sum(
                    torch.div(order[:count], blockcount, rounding_mode='trunc') == i
                ).item()
                losses[j] += torch.sum(self.Losses[i, :(perrow + 1)]).item()
                Ws[j][i, :] = Trace[perrow, i % parallel, :]
        for sparsity, loss in zip(sparsities, losses):
            print('%.4f error' % sparsity, loss)
            if DEBUG:
                tmp = self.layer.weight.data.clone()
                self.layer.weight.data = Ws[sparsities.index(sparsity)].reshape(self.layer.weight.shape) 
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2) / 128)
                self.layer.weight.data = tmp
        return Ws

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


DEFAULT_SPARSITIES = []
MIN_SPARSITY = 0
MAX_SPARSITY = .99
DELTA_SPARSE = .1
density = 1 - MIN_SPARSITY 
while density > 1 - MAX_SPARSITY:
    DEFAULT_SPARSITIES.append(1 - density)
    density *= 1 - DELTA_SPARSE 
DEFAULT_SPARSITIES.append(MAX_SPARSITY)

def gen_database(
    sparsedir,
    get_model, run, dataloader,
    dataloader_passes=1,
    sparsities=DEFAULT_SPARSITIES
):
    modelp = get_model()
    modeld = get_model()
    layersp = find_layers(modelp)
    layersd = find_layers(modeld)

    sds = {s: copy.deepcopy(modelp).cpu().state_dict() for s in sparsities}

    trueobs = {}
    for name in layersp:
        layer = layersp[name]
        trueobs[name] = TrueOBS(layer)

    cache = {}
    def add_batch(name):
        def tmp(layer, inp, out):
            trueobs[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in trueobs:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for i in range(dataloader_passes):
        for j, batch in enumerate(dataloader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()

    for name in trueobs:
        print(name)
        if name != 'conv1':
            continue
        print('Unstructured pruning ...')
        trueobs[name].prepare_unstr()
        Ws = trueobs[name].prune_unstr(sparsities)
        for sparsity, W in zip(sparsities, Ws):
            sds[sparsity][name + '.weight'] = W.reshape(sds[sparsity][name + '.weight'].shape).cpu()
        trueobs[name].free()

    for sparsity in sparsities:
        name = '%s_%04d.pth' % (args.model, int(sparsity * 10000))
        torch.save(sds[sparsity], os.path.join(sparsedir, name))


if __name__ == '__main__':
    import argparse

    from datautils import *
    from modelutils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('sparsedir', type=str)

    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=1024)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--noaug', action='store_true')

    args = parser.parse_args()

    dataloader, testloader = get_loaders(
        args.dataset, path=args.datapath,
        nsamples=args.nsamples, seed=args.seed,
        noaug=args.noaug
    )
    get_model, test, run = get_functions(args.model)

    os.makedirs(args.sparsedir, exist_ok=True)
    gen_database(
        args.sparsedir,
        get_model, run, dataloader,
        args.rounds
    )
