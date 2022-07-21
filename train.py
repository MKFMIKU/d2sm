#!/usr/bin/env python
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from datasets import create_dataset
from models import create_model
from option import parser

args = parser.parse_args()


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0 and not args.debug:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # create model
    print("=> creating model '{}'".format(args.model))
    model = create_model(args)
    model.print_networks(verbose=False)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset = create_dataset(args, training=True)
    eval_dataset = create_dataset(args, training=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        writer = SummaryWriter(log_dir="runs/" + args.name)
    else:
        writer = None

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, epoch, writer, args)

        if epoch % args.test_freq == 0:
            eval(eval_loader, model, epoch, writer, args)
        torch.cuda.empty_cache()

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            checkpoint_dir = os.path.join(args.checkpoint, args.name)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": args.model,
                    "state_dict": model.state_dict(),
                },
                is_best=False,
                filename=os.path.join(
                    checkpoint_dir, "checkpoint_{:04d}.pth.tar".format(epoch)
                ),
            )


def train(train_loader, model, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    lr = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        [batch_time, data_time, lr, losses], prefix="Training: [{:03d}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        global_step = i + epoch * len(train_loader)

        # unpack data and calculate loss functions
        model.module.set_input(batch)
        loss, batch_size = model.module.optimize_parameters(global_step)

        losses.update(loss.item(), batch_size)
        lr.update(model.module.get_lr(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Avoid Muliple Log
        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed
            and args.rank % torch.cuda.device_count() == 0
        ):
            if global_step % args.print_freq == 0:
                progress.display(global_step)
                writer.add_scalar("lr", model.module.get_lr(), global_step)
                writer.add_scalar("epoch", epoch, global_step)
                for k, v in model.module.get_current_losses().items():
                    writer.add_scalar("loss/%s" % k, v, global_step)
                writer.add_scalar("final_loss", loss, global_step)

            if global_step % args.display_freq == 0:
                for k, v in model.module.get_current_visuals().items():
                    writer.add_images("visual_%s" % k, v, global_step)
    # Epoch Based Learing Rate Optimizer
    model.module.scheduler.step()


def eval(test_loader, model, epoch, writer, args):
    batch_time = AverageMeter("Time", ":6.3f")
    avg_acc = AverageMeter("ACC", ":6.3f")
    progress = ProgressMeter([batch_time, avg_acc], prefix="Evaling: ")

    model.eval()
    end = time.time()
    for i, batch in enumerate(test_loader):
        model.module.set_input(batch)

        with torch.no_grad():
            acc = model.module.eval()

        avg_acc.update(acc)
        batch_time.update(time.time() - end)
        end = time.time()

    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % torch.cuda.device_count() == 0
    ):
        progress.display(epoch)
        writer.add_scalar("ACC", avg_acc.val, epoch)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + "[{:07d}]".format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
