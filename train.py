import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time


def eval(model, data_loader, device, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, count = 0, 0
    with torch.no_grad():
        len_data_loader = len(data_loader)
        iterator = iter(data_loader)
        tot_bi_loss = 0.
        tot_ce_loss = 0.
        for i in range(1, len_data_loader + 1):
            # Get mini-batch data
            data_pair = iterator.next()
            data, label = data_pair[0].to(device).float(), data_pair[1].to(device)

            # Forward propagation
            out = model.forward(data)

            # Compute loss values
            ce_loss = criterion(out, label.long())
            tot_ce_loss += ce_loss.item() / len_data_loader

            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            count += pred.shape[0]

    accuracy = float(correct) / float(count) * 100.
    return accuracy, tot_ce_loss


def train(model, train_data_loader_w, train_data_loader_a, test_data_loader, device, args, is_arch_learning=False):
    criterion = nn.CrossEntropyLoss()
    # weight_optimizer, arch_optimizer, weight_param_list, arch_param_list\
    #     = get_optimizers(model, args)
    weight_params = []
    arch_params = []
    for _n, _p in model.named_parameters():
        if 'alpha' in _n:
            arch_params.append(_p)
        else:
            weight_params.append(_p)

    weight_optimizer = optim.SGD(weight_params, lr=args.wlr, weight_decay=5e-4, momentum=0.9)
    arch_optimizer = optim.SGD(arch_params, lr=args.alr)
    # if args.scheduler == 'cosine':
    #     cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(weight_optimizer, T_max=args.max_epoch)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        weight_optimizer,
        milestones=[80, 120],
        gamma=0.1
    )
    # Lists for plots
    train_ce_losses = []

    min_loss = 1e10
    final_best_model = model
    final_best_epoch = -999
    final_ce_loss = 1e10

    for epoch in range(1, args.max_epoch + 1):
        epoch_st = time.time()
        # model.update_drop_path_prob(args.drop_path * epoch / args.max_epoch)
        if is_arch_learning:
            data_loader = train_data_loader_a
        else:
            data_loader = train_data_loader_w
        model.train()
        tot_ce_loss = 0.
        len_data_loader = len(data_loader)
        iterator = iter(data_loader)
        for i in range(1, len_data_loader + 1):
            # model.zero_grad()
            weight_optimizer.zero_grad()
            data_pair = iterator.next()
            data, label = data_pair[0].to(device).float(), data_pair[1].to(device)

            # Forward propagation
            out = model(data)

            # Compute loss values
            ce_loss = criterion(out, label.long())
            tot_ce_loss += ce_loss.item() / len_data_loader
            loss = ce_loss
            loss.backward()
            if is_arch_learning:
                arch_optimizer.step()
            else:
                weight_optimizer.step()

        # Adjusting learning rate if needed
        if not is_arch_learning:
            lr_scheduler.step()

        train_acc, train_ce_loss = eval(model, train_data_loader_w, device, args)
        print("Epoch: {:3}/ Train acc: {:.3f}, loss: {:.3f} " \
              .format(epoch, train_acc, train_ce_loss), end='/ ')
        test_acc, test_ce_loss = eval(model, test_data_loader, device, args)
        print("Test acc: {:.3f}/ loss: {:.3f}" \
              .format(test_acc, test_ce_loss), end='/ ')
        print(f"{time.time()-epoch_st:.2f}s")

        ce_loss_epoch = train_ce_loss


        # Keep the all-time best model
        if ce_loss_epoch < final_ce_loss:
            final_ce_loss = ce_loss_epoch
            final_best_model = pickle.loads(pickle.dumps(model))
            final_best_epoch = epoch

        # Add values for plotting
        train_ce_losses.append(tot_ce_loss)
        # valid_ce_losses.append(valid_ce_loss)
        # valid_bi_losses.append(valid_bi_loss * lam)
        # valid_accuracies.append(valid_acc)
    return final_best_model, final_best_epoch, final_ce_loss
