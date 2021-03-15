from pathlib import Path

import torch
from tqdm import tqdm
from scipy.io import savemat

from ..data import MAGICDatasetZpad
from ..data.loraks import MAGICDatasetLORAKS
from ..net import JVSNet


def setup(parser):
    parser.add_argument("--name", type=str, help="Name")
    parser.add_argument("--weight-file", type=str, help="Path for model weight")
    parser.add_argument(
        "--test-file", action="append", type=str, help="Path for input h5 testing"
    )
    parser.add_argument(
        "--zpad", type=bool, default=False, help="Zero_pad as input or not"
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument(
        "--cascades", type=int, default=8, help="Cascade number for network"
    )
    parser.set_defaults(func=main)


def main(args):
    valtest_aug = dict(
        augment_flipud=False,
        augment_fliplr=False,
        augment_scale=False,
    )

    model = JVSNet(num_echos=30, alfa=0.1, beta=0.1, cascades=args.cascades)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    loc = "cuda:{}".format(args.gpu)
    data = torch.load(args.weight_file, map_location=loc)
    state_dict = {k[len("module.") :]: v for k, v in data["state_dict"].items()}
    model.load_state_dict(state_dict)
    print(f"Loaded weight at epoch {data['epoch']}")

    if args.zpad:
        print("input is from Zero-Padding")
        db_class = MAGICDatasetZpad
    else:
        print("input is from LORAKS")
        db_class = MAGICDatasetLORAKS

    datasets = {
        Path(f).stem: db_class(f, **valtest_aug, verbosity=False)
        for f in args.test_file
    }

    to_save = {}
    for fname, dataset in tqdm(datasets.items()):
        output, gt = evaluate(dataset, model, args.gpu)
        to_save[fname] = dict(output=output, gt=gt)

    savemat(f"{args.name}_result.mat", to_save)


def evaluate(dataset, model, gpu):
    model.eval()

    outputs = []
    gts = []

    with torch.no_grad():
        for im0, true, tsens, tx_kjvc, tmask in tqdm(dataset):
            im0 = im0.unsqueeze(0).cuda(gpu, non_blocking=True)
            true = true.unsqueeze(0).cuda(gpu, non_blocking=True)
            tsens = tsens.unsqueeze(0).cuda(gpu, non_blocking=True)
            tx_kjvc = tx_kjvc.unsqueeze(0).cuda(gpu, non_blocking=True)
            tmask = tmask.unsqueeze(0).cuda(gpu, non_blocking=True)

            output = model(im0, tx_kjvc, tmask, tsens)

            outputs += [output.detach().clone().cpu()]
            gts += [true.detach().clone().cpu()]

    return torch.cat(outputs).numpy(), torch.cat(gts).numpy()
