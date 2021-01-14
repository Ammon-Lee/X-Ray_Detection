from src.ssd_model import SSD300, Backbone
import torch
import transform
from my_dataset import XRayDataset
import os
import train_utils.train_eval_utils as utils
from train_utils.coco_utils import get_coco_api_from_dataset


def create_model(num_classes=21, device=torch.device('cpu')):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    # pre_train_path = "./src/resnet50.pth"
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    # https://ngc.nvidia.com/catalog/models -> search ssd -> download FP32
    pre_ssd_path = "./save_weights/ssd300-23.pth"
    pre_model_dict = torch.load(pre_ssd_path, map_location=device)
    pre_weights_dict = pre_model_dict["model"]

    # Remove category predictor weights
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


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

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

    XRay_root = parser_data.data_path
    train_dataset = XRayDataset(XRay_root, data_transform['train'], train_set='train.txt')
    # Note that the batch_size must be greater than 1
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=utils.collate_fn)

    val_dataset = XRayDataset(XRay_root, data_transform['val'], train_set='val.txt')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.collate_fn)

    model = create_model(num_classes=6, device=device)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.3)

    # If the address of the weight file saved by the last training is specified, the training continues with the last result
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    val_data = None
    # If your computer has sufficient memory, you can save time by loading the validation set data in advance to avoid having to reload the data each time you validate
    # val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        utils.train_one_epoch(model=model, optimizer=optimizer,
                              data_loader=train_data_loader,
                              device=device, epoch=epoch,
                              print_freq=50, train_loss=train_loss,
                              train_lr=learning_rate)

        lr_scheduler.step()

        utils.evaluate(model=model, data_loader=val_data_loader,
                       device=device, data_set=val_data, mAP_list=val_map)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/ssd300-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)

    # inputs = torch.rand(size=(2, 3, 300, 300))
    # output = model(inputs)
    # print(output)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # Type of training equipment
    parser.add_argument('--device', default='cuda:0', help='device')
    # The root directory of the training dataset
    parser.add_argument('--data-path', default='./', help='dataset')
    # File save address
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # If you need to follow up the last training, specify the address of the weight file saved for the last training
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # Specifies the epoch number from which training begins next
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # The total epoch number of training
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    print(args)

    # Check whether the save weight folder exists, create if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
