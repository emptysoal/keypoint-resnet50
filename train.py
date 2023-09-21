# -*- coding: utf-8 -*-

import os
import logging
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from kpt_resnet50 import KptResNet50
from kpt_hrnet import PoseHighResolutionNet
from dataset import KeypointDataset
from evaluate import accuracy


def get_args():
    parser = argparse.ArgumentParser(description="UNet Keypoint Detection")
    # model construction
    parser.add_argument('--kpt-num', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--heatmap-size', type=int, default=112)
    parser.add_argument('--sigma', type=float, default=3.0)
    # training hyper params
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument("--weight-decay", type=float, default=0.0008)
    # data and model path
    parser.add_argument("--data-dir", type=str, default="./data/car_plate")
    parser.add_argument("--model-save-dir", type=str, default="./model/")
    parser.add_argument("--save-interval", type=int, default=10)

    return parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    train_loss = 0.0
    train_acc = 0.0
    tbar = tqdm(train_loader)
    for step, (image, target) in enumerate(tbar):
        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        tbar.set_description('Train loss: %.6f' % (train_loss / (step + 1)))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        train_acc += avg_acc

    avg_loss = train_loss / len(train_loader)
    avg_accuracy = train_acc / len(train_loader)

    logging.info('Epoch: %d train done, Loss: %.6f, Accuracy: %.4f%%.' % (epoch, avg_loss, avg_accuracy * 100))

    return avg_loss, avg_accuracy


def validate(model, device, valid_loader, criterion, epoch):
    # switch to evaluate mode
    model.eval()

    valid_loss = 0.0
    valid_acc = 0.0
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for step, (image, target) in enumerate(tbar):
            image, target = image.to(device), target.to(device)
            output = model(image)

            loss = criterion(output, target)
            valid_loss += loss.item()
            tbar.set_description('Test loss: %.6f' % (valid_loss / (step + 1)))

            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            valid_acc += avg_acc

    avg_loss = valid_loss / len(valid_loader)
    avg_accuracy = valid_acc / len(valid_loader)

    logging.info('Epoch: %d valid done, Loss: %.6f, Accuracy: %.4f%%.' % (epoch, avg_loss, avg_accuracy * 100))

    return avg_loss, avg_accuracy


def run():
    train_dataset = KeypointDataset(args, split="train")
    valid_dataset = KeypointDataset(args, split="val")

    kwargs = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    model = KptResNet50(kpt_num=args.kpt_num, pretrained=True)
    # model = PoseHighResolutionNet(kpt_num=args.kpt_num, pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-9)
    criterion = torch.nn.MSELoss()

    logging.info("Start training.")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d start." % epoch)
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_acc = validate(model, device, valid_loader, criterion, epoch)
        scheduler.step()

        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Valid/Loss", valid_loss, epoch)
        writer.add_scalar("Valid/Accuracy", valid_acc, epoch)

        if args.save_interval != -1 and epoch % args.save_interval == 0:
            save_path = os.path.join(args.model_save_dir, "model_%s.pth" % epoch)
            torch.save(model.state_dict(), save_path)

        if valid_acc > best_acc:
            best_save_path = os.path.join(args.model_save_dir, "best.pth")
            torch.save(model.state_dict(), best_save_path)
            best_acc = valid_acc
            logging.info("Succeeded saving best.pth, and valid accuracy is %.4f%%" % (best_acc * 100))
        logging.info("Previous best accuracy is : %.4f%%" % (best_acc * 100))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    logging.info(args)

    writer = SummaryWriter(args.model_save_dir)

    run()
