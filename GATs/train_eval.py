import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from load_data import load_data

# from torch_geometric.utils import index_to_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1

    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def run(args, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None, kl_loss=False, kl_alpha1=0.01, kl_alpha2=0.02):
    val_losses, accs, durations = [], [], []
    print("RUN ARGS: {}".format(args))
    for _ in range(runs):

        args.seed += _
        data = load_data(args)

        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data, kl_loss, kl_alpha1, kl_alpha2)
            eval_info = evaluate(model, data, kl_loss)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print(f'Val Loss: {float(loss.mean()):.4f}, '
          f'Test Accuracy: {float(acc.mean()):.3f} ± {float(acc.std()):.3f}, '
          f'Duration: {float(duration.mean()):.3f}')

    return loss, acc, duration


def train(model, optimizer, data, kl_loss, kl_alpha1, kl_alpha2):
    model.train()
    optimizer.zero_grad()
    if kl_loss is True:
        out, noise1, noise2 = model(data)
        # kl_loss for noise
        y_uni = torch.ones_like(noise2) / (torch.max(data.y) + 1)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) \
               + kl_alpha1 * compute_kl_loss(noise1, y_uni) \
               + kl_alpha2 * compute_kl_loss(noise2, y_uni)
    else:
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data, kl_loss):
    model.eval()

    with torch.no_grad():
        if kl_loss is True:
            logits, noise1, noise2 = model(data)
        else:
            logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data[f'{key}_mask']
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs[f'{key}_loss'] = loss
        outs[f'{key}_acc'] = acc

    return outs


# same from your code, with 'batchmean' change to 'mean'
def compute_kl_loss(x, y):
    return F.kl_div(F.log_softmax(x, dim=-1), F.softmax(y, dim=-1), reduction='mean')
