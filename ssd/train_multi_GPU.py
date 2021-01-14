import torch
import train_utils.train_eval_utils as utils
import time
import os
import datetime
from my_dataset import VOC2012DataSet
from train_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from src.ssd_model import SSD300, Backbone
import transform
import torch.multiprocessing as mp


def create_model(num_classes, device=torch.device('cpu')):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    pre_train_path = "./src/resnet50.pth"
    backbone = Backbone(pretrain_path=pre_train_path)
    model = SSD300(backbone=backbone, num_classes=num_classes)

    pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"
    pre_model_dict = torch.load(pre_ssd_path, map_location=device)
    pre_weights_dict = pre_model_dict["model"]

    # delete weights of prediction categories
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model


# def main_worker(args):
def main(args):
    print(args)
    # mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    data_transform = {
        "train": transform.Compose([transform.SSDCropping(),
                                    transform.Resize(),
                                    transform.ColorJitter(),
                                    transform.ToTensor(),
                                    transform.RandomHorizontalFlip(),
                                    transform.Normalization(),
                                    transform.AssignGTtoDefaultBox()]),
        "val": transform.Compose([transform.Resize(),
                                  transform.ToTensor(),
                                  transform.Normalization()])
    }

    VOC_root = args.data_path
    # load train data set
    train_data_set = VOC2012DataSet(VOC_root, data_transform["train"], train_set='train.txt')

    # load validation data set
    val_data_set = VOC2012DataSet(VOC_root, data_transform["val"], train_set='val.txt')

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data_set)
        test_sampler = torch.utils.data.SequentialSampler(val_data_set)

    if args.aspect_ratio_group_factor >= 0:
        # count all scales of images in position index in bins.
        group_ids = create_aspect_ratio_groups(train_data_set, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_data_set, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_data_set, batch_size=4,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = create_model(num_classes=21)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # If resume arg is not none, the training will continue after the resume arg.
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # Read previously saved weight files (including optimizer and learning rate policy)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            # Save weight operations are performed only on the primary node.
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        utils.evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    # The root of the training file
    parser.add_argument('--data-path', default='./', help='dataset')
    # Type of training equipment
    parser.add_argument('--device', default='cuda', help='device')
    # Batch_size per GPU
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # Specifies the epoch number from which training begins next
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # The total epoch number of training
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    # Number of threads of data loading and preprocessing
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Learning rate, which needs to be set according to the number of GPUs and batch_size 0.005/8 * num_GPU
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.005 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    # momentum of SGD
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # weight_decay of SGD
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Parameters for torch.optim.lr_scheduler.StepLR
    parser.add_argument('--lr-step-size', default=5, type=int, help='decrease lr every step-size epochs')
    # Parameters to the torch.optim.lr_scheduler. MultiSteplr
    parser.add_argument('--lr-steps', default=[7, 12], nargs='+', type=int, help='decrease lr every step-size epochs')
    # Parameters to the torch.optim.lr_scheduler. MultiSteplr
    parser.add_argument('--lr-gamma', default=0.3, type=float, help='decrease lr by a factor of lr-gamma')
    # The frequency at which information is printed during training
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # File save address
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    # Continue training based on the results of the last training session
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # No training, just testing
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Number of processes started (not threads)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    # If the save file address is specified, check to see if the folder exists. If it does not, create it
    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
