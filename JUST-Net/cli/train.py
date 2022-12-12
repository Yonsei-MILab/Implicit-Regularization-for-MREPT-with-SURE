import os
import random
import shutil

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import numpy as np
import pkbar

from ..net import JVSNet
from ..data import MAGICDatasetZpad
from ..data.dataset import MakeDataset
from ..utils import complex_abs


best_loss = 1e12


def setup(parser):
    parser.add_argument("--name", type=str, help="Name")
    parser.add_argument("--train-file", type=str, help="Path for input h5 training")
    parser.add_argument("--val-file", type=str, help="Path for input h5 validation")
    parser.add_argument("--test-file", type=str, help="Path for input h5 testing")
    parser.add_argument(
        "--zpad", type=bool, default=False, help="Zero_pad as input or not"
    )
    parser.add_argument(
        "--cascades", type=int, default=10, help="Cascade number for network"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of Batchsize"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Epoch number for training"
    )
    parser.add_argument(
        "--learning-rate", default=3e-4, type=float, help="Learning rate for network"
    )
    parser.add_argument(
        "--no-augment-flipud",
        type=bool,
        default=False,
        help="No augmentation while Training - Flip UD",
    )
    parser.add_argument(
        "--no-augment-fliplr",
        type=bool,
        default=False,
        help="No augmentation while Training - Flip LR",
    )
    parser.add_argument(
        "--no-augment-scale",
        type=bool,
        default=False,
        help="No augmentation while Training - Scale",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--workers",
        default=0,  # TODO: Fix workers
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--plot-index-val",
        default=64,
        type=int,
        metavar="I",
        help="plot index I of validation dataset",
    )
    parser.add_argument(
        "--plot-index-test",
        default=64,
        type=int,
        metavar="I",
        help="plot index I of test dataset",
    )
    parser.add_argument(
        "--plot-freq",
        default=1,
        type=int,
        metavar="N",
        help="plot every N epoch",
    )
    parser.set_defaults(func=main)


def main(args):
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))


def main_worker(gpu, ngpus, args):
    global best_loss

    print("Use GPU: {} for training".format(gpu))

    rank = gpu
    world_size = ngpus
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:13345",
        world_size=world_size,
        rank=rank,
    )

    model = JVSNet(num_echos=14, alfa=0.1, beta=0.1, cascades=args.cascades)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # for n, p in model.named_parameters():
    #     print(n, p.size())

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus)
    args.workers = int((args.workers + ngpus - 1) / ngpus)
    model = DistributedDataParallel(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # print("=> loading checkpoint '{}'".format(args.resume))
            # loc = "cuda:{}".format(gpu)
            loc = "cpu"
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    torch.backends.cudnn.benchmark = True

    print((
            "Augmentation Configuration:\n"
            f"\tFlipUD: {not args.no_augment_flipud}\n"
            f"\tFlipLR: {not args.no_augment_fliplr}\n"
            f"\tScale: {not args.no_augment_scale}"
        )
    )

    train_aug = dict(
        augment_flipud=not args.no_augment_flipud,
        augment_fliplr=not args.no_augment_fliplr,
        augment_scale=not args.no_augment_scale,
    )
    
    valtest_aug = dict(
        augment_flipud=False,
        augment_fliplr=False,
        augment_scale=False,
    )

    # Data loading code
    if args.zpad:
        print("input is from Zero-Padding")
        train_dataset = MAGICDatasetZpad(args.train_file, **train_aug, verbosity=False)
        val_dataset = MAGICDatasetZpad(args.val_file, **valtest_aug, verbosity=False)
        test_dataset = MAGICDatasetZpad(args.test_file, **valtest_aug, verbosity=False)
    else:
        # print("input is from dataset.py")
        train_dataset = MakeDataset(
            args.train_file, **train_aug, verbosity=False
        )
        val_dataset = MakeDataset(args.val_file, **valtest_aug, verbosity=False)
        test_dataset = MakeDataset(
            args.test_file, **valtest_aug, verbosity=False
        )
    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # difference
        num_workers=args.workers,  # difference
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    valdata = val_dataset[args.plot_index_val]
    testdata = test_dataset[args.plot_index_test]

    is_printer = rank % ngpus == 0
    if is_printer:
        writer = SummaryWriter(f"runs/{args.name}")

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        kbar = (
            pkbar.Kbar(
                target=len(train_loader), epoch=epoch, num_epochs=args.epochs, width=8
            )
            if is_printer
            else None
        )

        # train for one epoch
        train_loss, _ = train(train_loader, model, criterion, optimizer, gpu, kbar)

        # evaluate on validation set
        val_loss, val_rmse = validate(val_loader, model, criterion, gpu)
        if kbar is not None:
            kbar.add(1, values=[("val_loss", val_loss), ("val_rmse", val_rmse)])

        # remember best loss and save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if is_printer:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.name,
            )

            writer.add_scalar("Loss/training", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)

            if epoch % args.plot_freq == 0:
                valimage = show_output(model, valdata, gpu)
                testimage = show_output(model, testdata, gpu)
                writer.add_image(
                    f"validation_index{args.plot_index_val}", valimage, epoch
                )
                writer.add_image(f"test_index{args.plot_index_test}", testimage, epoch)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.detach().clone().cpu(), epoch)

    if is_printer:
        writer.close()

    dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, gpu, kbar):
    model.train()

    losses = []
    rmses = []

    for i, (im0, true, tsens, tx_kjvc, tmask) in enumerate(train_loader):
        im0 = im0.cuda(gpu, non_blocking=True)
        true = true.cuda(gpu, non_blocking=True)
        tsens = tsens.cuda(gpu, non_blocking=True)
        tx_kjvc = tx_kjvc.cuda(gpu, non_blocking=True)
        tmask = tmask.cuda(gpu, non_blocking=True)

        # compute output
        output = model(im0, tx_kjvc, tmask, tsens)
        loss = criterion(output, true)
        rmse = torch.sqrt(loss)

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += [loss.item()]
        rmses += [rmse.item()]

        if kbar is not None:
            kbar.update(i, values=[("loss", loss.item()), ("rmse", rmse.item())])


    return sum(losses) / len(losses), sum(rmses) / len(rmses)


def validate(val_loader, model, criterion, gpu):
    model.eval()

    losses = []
    rmses = []

    with torch.no_grad():
        for im0, true, tsens, tx_kjvc, tmask in val_loader:
            im0 = im0.cuda(gpu, non_blocking=True)
            true = true.cuda(gpu, non_blocking=True)
            tsens = tsens.cuda(gpu, non_blocking=True)
            tx_kjvc = tx_kjvc.cuda(gpu, non_blocking=True)
            tmask = tmask.cuda(gpu, non_blocking=True)

            # compute output
            output = model(im0, tx_kjvc, tmask, tsens)
            loss = criterion(output, true)
            rmse = torch.sqrt(loss)

            losses += [loss.item()]
            rmses += [rmse.item()]

    return sum(losses) / len(losses), sum(rmses) / len(rmses)


def save_checkpoint(state, is_best, name):
    filename = f"{name}_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{name}_best.pth.tar")


def show_output(model, testdata, gpu):
    model.eval()

    with torch.no_grad():
        im0, true, tsens, tx_kjvc, tmask = testdata
        im0 = im0.unsqueeze(0).cuda(gpu, non_blocking=True)
        true = true.unsqueeze(0).cuda(gpu, non_blocking=True)
        tsens = tsens.unsqueeze(0).cuda(gpu, non_blocking=True)
        tx_kjvc = tx_kjvc.unsqueeze(0).cuda(gpu, non_blocking=True)
        tmask = tmask.unsqueeze(0).cuda(gpu, non_blocking=True)

        out = model(im0, tx_kjvc, tmask, tsens)

    abs_tensor1 = complex_abs(im0.detach().cpu())
    abs_tensor2 = complex_abs(out.detach().cpu())
    abs_tensor3 = complex_abs(true.detach().cpu())
    abs_tensor4 = complex_abs(out.detach().cpu() - true.detach().cpu()) * 20

    def _make_image(idx):
        return make_grid(
            [
                abs_tensor1[0, idx].unsqueeze(0),
                abs_tensor2[0, idx].unsqueeze(0),
                abs_tensor3[0, idx].unsqueeze(0),
                abs_tensor4[0, idx].unsqueeze(0),
            ],
            normalize=True,
            range=(0, 1.5),
            padding=0,
        )

    return make_grid([_make_image(i) for i in range(14)], nrow=1, padding=0)
