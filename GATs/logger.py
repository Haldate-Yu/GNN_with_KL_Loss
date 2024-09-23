# @Time     : 2022/11/9
# @Author   : Haldate
import os


def logger(model_name, loss, acc, duration, args):
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/{}'.format(args.data), exist_ok=True)
    file = './results/{}/{}.txt'.format(args.data, model_name)

    StartLine = '=========='
    BaseParams = 'Params:  lr={}, wd={}, epochs={}, hidden={}'.format(args.lr, args.weight_decay, args.epochs,
                                                                      args.hidden)
    if args.kl_loss is True:
        KLparams = 'KL Params:  mlp hidden={}, alpha1={}, alpha2={}, attn topk={}'.format(args.hidden_mlp,
                                                                                          args.kl_alpha1,
                                                                                          args.kl_alpha2,
                                                                                          args.attn_topk)
    Result = 'Val Loss: {:.4f}, \nTest Acc: {:.4f} ± {:.3f}, \nDuration: {}'.format(float(loss.mean()),
                                                                                    float(acc.mean()), float(acc.std()),
                                                                                    float(duration.mean()))
    with open(file, 'a+') as f:
        if args.kl_loss is True:
            f.write('{}\n {}\n {}\n {}\n'.format(StartLine, BaseParams, KLparams, Result))
        else:
            f.write('{}\n {}\n {}\n'.format(StartLine, BaseParams, Result))


def csv_logger(model_name, loss, acc, duration, args):
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/{}'.format(args.dataset), exist_ok=True)
    file = './results/{}/{}.csv'.format(args.dataset, model_name)

    headerList = [
        'Method', 'Dataset',
        # Base Params
        'Learning Rate', 'Weight Decay', 'Epochs', 'Hidden Dims',
        # KL Params
        'MLP Dims', 'alpha1', 'alpha2', 'Attn TopK',
        # Results
        'Val Loss', 'Test Acc', 'Duration'
    ]

    with open(file, 'a+') as f:
        # csv reader verification
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',', fieldnames=headerList)
            dw.writeheader()
        content = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.4f}, {:.4f} ± {:.3f}, {}\n".format(
            model_name, args.dataset,
            args.lr, args.weight_decay, args.epochs, args, hidden,
            args.hidden_mlp, args.kl_alpha1, args.kl_alpha2, args.attn_topk,
            float(loss.mean()), float(acc.mean()), float(acc.std()), float(duration.mean())
        )
        f.write(content)
