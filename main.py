import os, sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import torch.utils.data
import torch.nn as nn

import time, datetime
import random
import pickle

from TreeNeuralNet import TreeNeuralNet
import torchvision.datasets as dset

from train import train, eval
import utils
import parser

def main():
    # Load arguments and set environment
    yaml_file = './config/cifar10.yaml'
    args = parser.get_parser_from_yaml(yaml_file)

    save_pth = args.res_path
    checkpt_pth = save_pth + '/models'

    if not os.path.exists(checkpt_pth):
        os.makedirs(checkpt_pth)
        print("Make directory: ", checkpt_pth)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    # Load CIFAR-10 dataset
    num_classes = 10
    train_transform, test_transform = utils.data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

    num_train = len(train_data)
    indices = list(range(num_train))

    load_model = False
    if load_model:
        # Load model
        with open('.pth', 'rb') as f:
            l = pickle.load(f)
        model = l['model']
        args = l['args']
        is_arch_learning = True
        fined_tuned = 1
        device = 'cpu'
    else:
        # Generate model
        model = TreeNeuralNet(args.c, args.max_depth, args.borders,
                              args.ops, num_classes)
        is_arch_learning = False
        fine_tuned=0

    if len(args.gpus.split(',')) > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # if args.batch == -1:
    #     args.batch = utils.adjust_batch_size(model, init_bs=1000, data_shape=(3, 32, 32), device=device)
    #     print(args.batch)

    train_data_loader_w = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch,
        pin_memory=True, num_workers=1)

    train_data_loader_a = torch.utils.data.DataLoader(
        train_data, batch_size=500,
        pin_memory=True, num_workers=1)

    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch,
        pin_memory=True, num_workers=2)

    print('==================================================')
    print(f"Start training at {datetime.datetime.now()}")
    num_params = utils.number_of_params(model)
    print("# Params: ", num_params)
    print("================================================")
    st = time.time()
    model, best_epoch, valid_loss = train(model, train_data_loader_w,
                                          train_data_loader_a,
                                          test_data_loader, device, args,
                                          is_arch_learning=is_arch_learning)
    tt = time.time() - st
    hh, mm, ss = tt // 3600, (tt % 3600) // 60, tt % 60
    print(f"Training ends at {datetime.datetime.now()}")
    print(f"Spent time: {hh}H {mm}m {ss}s")
    print("# Params: ", num_params)
    print("Final CE loss: ", valid_loss)

    # TODO: Test trained model
    train_accuracy_alpha, _ = eval(model, train_data_loader_w, device, args)
    accuracy_alpha, bce_loss_alpha = eval(model, test_data_loader, device, args)
    print(f"Train accuracy (w/ alpha): {train_accuracy_alpha:.4f}")
    print(f"Test accuracy (w/ alpha): {accuracy_alpha:.4f} [Best Epoch: {best_epoch}]")

    # edge_stats = model.get_edge_stats(args)
    # print(f"Active: {edge_stats['active']}, Inactive: {edge_stats['inactive']}")

    ###### End of training. Save model and write the result ######

    # Save model package
    # save model with its parser for further analysis
    model_package = {
        'model': model.to('cuda:0'),
        'args': args
    }

    with open(checkpt_pth+f'/model_{args.num}_{fine_tuned}.pth', 'wb') as f:
        pickle.dump(model_package, f)

    # Write result
    with open(os.path.join(args.res_path, 'result.txt'), 'a') as f:
        res = ''
        for arg in vars(args):
            if arg in ['result_path', 'data_path']:
                continue
            res += f"{getattr(args, arg)}, "
        f.write(res + "{},{},{},{},{}\n"
                .format(accuracy_alpha, accuracy_alpha, num_params, valid_loss, best_epoch))


if __name__ == "__main__":
    main()