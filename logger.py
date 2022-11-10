# @Time     : 2022/11/9
# @Author   : Haldate
import os


def logger(model_name, loss, acc, duration, args):
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/{}'.format(args.dataset), exist_ok=True)
    file = './results/{}/{}.txt'.format(args.dataset, model_name)

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
