import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import zip_longest, chain
import torch
from util.misc import assert_shape
from inspect import signature
import time
from torch import optim
from util.pose import mrpe


def exp_decay(params):
    def f(epoch):
        return params.learning_rate * (0.96 ** (epoch * 0.243))

    return f


def dataset2numpy(dataset, fields):
    """
    Converts a PyTorch Dataset to a numpy array.

    Parameters:
        fields: list of fields to return from the full dataset.
    """

    loader = DataLoader(dataset, batch_size=len(dataset) // 8, num_workers=8)
    parts = []
    for l in loader:
        parts.append(l)

    return [np.concatenate([p[f].numpy() for p in parts], axis=0) for f in fields]


def torch_predict(model, input, batch_size=None, device='cuda'):
    """

    :param model: PyTorch Model(nn.Module)
    :param input: a numpy array or a PyTorch dataloader
    :param batch_size: if input was a numpy array, this is the batch size used for evaluation
    :return:
    """
    model.eval()

    if isinstance(input, np.ndarray):
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input).to(device)), batch_size)
        needs_move = False
    elif isinstance(input, torch.Tensor):
        data_loader = DataLoader(TensorDataset(input.to(device)), batch_size)
        needs_move = False
    else:
        data_loader = input
        needs_move = True

    result = []
    with torch.no_grad():
        for batch in data_loader:
            if needs_move:
                if isinstance(batch, (list, tuple, map)):
                    batch = map(lambda x: x.to(device), batch)
                elif isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                else:
                    batch = batch.to(device)

            if isinstance(batch, (list, tuple, map)):
                pred = model(*batch)
            elif isinstance(batch, dict):
                pred = model(**batch)
            else:
                pred = model(batch)

            if isinstance(pred, (list, tuple, map)):
                result.append([x.cpu().numpy() for x in pred])
            else:
                result.append(pred.cpu().numpy())

            del pred

    if isinstance(result[0], list):
        out = []
        for i in range(len(result[0])):
            out.append(np.concatenate([x[i] for x in result]))
        result = out
    else:
        result = np.concatenate(result)

    return result


def torch_eval(model, loader, loss_fn, input_name, target_name, device='cuda'):
    """
    Evaluates a PyTorch model.

    :param model: PyTorch Model(nn.Module)
    :param loader: a PyTorch DataLoader producing input batches
    :param loss_fn: a function or dictionary of functions. The metrics evaluated. The functions
                    should return a single scalar torch tensor. They can have 3 parameters, the third is optional.
                    The first is the input to model, the second is the target variable, and the thirs is the full
                    batch used in the eval iteration.
                    It is expected that the output losses are averaged over the batch.
    :param input_name: name of input fields passed to the model (a single name or array of names)
    :return:
    """
    assert isinstance(loader, DataLoader)
    model.eval()

    loss_was_func = False
    if not isinstance(loss_fn, dict):
        loss_fn = {'loss': loss_fn}
        loss_was_func = True

    if not isinstance(input_name, (list, tuple)):
        input_name = [input_name]

    metrics = {}
    num_args = {}
    for name, func in loss_fn.items():
        metrics[name] = 0
        num_args[name] = len(signature(func).parameters)

    total_cnt = 0
    with torch.no_grad():
        for batch in loader:
            # batch = list(map(lambda x: x.to(device), batch))
            batch = {k: v.to(device) for k, v in batch.items()}

            pred = model(*[batch[x] for x in input_name])

            for name, loss_func in loss_fn.items():
                if num_args[name] == 2:
                    loss = loss_func(pred, batch[target_name])
                else:
                    loss = loss_func(pred, batch[target_name], batch)

                metrics[name] += loss.item() * len(batch[input_name[0]])

            total_cnt += len(batch[input_name[0]])

    for name in loss_fn.keys():
        metrics[name] /= total_cnt

    if loss_was_func:
        return metrics['loss']
    else:
        return metrics


def get_optimizer(parameters, config):
    if config['optimiser'] == "adam":
        return optim.Adam(parameters, lr=config['learning_rate'], amsgrad=config['adam_amsgrad'])
    elif config['optimiser'] == "rmsprop":
        return optim.RMSprop(parameters, lr=config['learning_rate'])
    elif config['optimiser'] == "sgd":
        return optim.SGD(parameters, lr=config['learning_rate'], momentum=config['sgd_momentum'])
    elif config['optimiser'] == "radam":
        return RAdam(parameters, lr=config['learning_rate'])
    else:
        raise Exception('Unimplemented optimiser: ' + config['optimiser'])


def _get_scheduler(optimizer, config):
    """ Decodes a scheduler config. Returns none if no schedulers were specified """
    if config is None or config['type'] == 'none':
        return None

    # scheduler = None
    # assert not _config['weight_decay'] or not _config['lr_div_10'], "weight decay and stepwise lr can't be turned on at the same time"
    if config['type'] == 'martinez_weight_decay':
        return optim.lr_scheduler.LambdaLR(optimizer, lambda x: (0.96 ** (x * 0.243)))
    elif config['type'] == 'multiplicative':
        return optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['multiplier'])
    elif config['type'] == 'lr_div_10_wd':  # exponential decay + division by ten at certain epochs
        def lr_fn(x):
            scale = config['lr_div_10_scale']
            base = (0.96 ** (x * 0.243))
            if x >= 80:
                factor = scale * scale
            elif x >= 40:
                factor = scale
            else:
                factor = 1

            return factor * base

        return optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    else:
        raise NotImplementedError("Unknown scheduler type: ", config['type'])



def torch_train(train_loader, model, update_fn, _config, callbacks=[]):
    """
    Trains a model.

    :param train_loader: training data is loaded from here, PyTorch DataLoader
    :param model: PyTorch model to train
    :param update_fn: the function called on every iteration, must calculate the loss
    :param _config: Sacred config object
    :param callbacks: optional callbacks for training
    :return:
    """
    optimizer = get_optimizer(model.parameters(), _config)

    scheduler = _get_scheduler(optimizer, _config['lr_scheduler'])

    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    epoch_len = _config['num_epochs']
    iter_cnt = 0
    for epoch in range(epoch_len):  # loop over the dataset multiple times
        model.train()

        epoch_loss = 0
        epoch_val = {}
        epoch_start = time.time()
        iter_start = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            batch_start = time.time()

            loss, vals = update_fn(model, data)

            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            for k, v in vals.items():
                epoch_val[k] = epoch_val.get(k, 0) + v

            del loss  # free up memory

            if (i + 1) % 50 == 0:  # print every 50 mini-batches
                iter_time = (time.time() - iter_start) / 50
                print('\r[%d, %5d] loss: %.3f b=%4dms i=%dms' % (epoch + 1, i + 1, running_loss / 50,
                                                                 int(batch_time * 1000), int(iter_time * 1000)), end='')
                for c in callbacks:
                    c.on_itergroup_end(iter_cnt, running_loss / 50)

                running_loss = 0.0
                iter_start = time.time()

            iter_cnt += 1

            if _config.get('SHORT_EPOCH', False):
                if i > 600:
                    break

        print("Iterations done:", i)
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_loss = epoch_loss / len(train_loader)
        epoch_val = {k: v / len(train_loader) for k, v in epoch_val.items()}
        print()
        print("Epoch %3d: loss: %4.3f   %4.1fs" % (epoch + 1, epoch_loss, epoch_time))

        # evaluate
        model.eval()
        for c in callbacks:
            c.on_epoch_end(model, epoch, epoch_loss, optimizer, epoch_val)


def set_requires_grad(module, requires_grad):
    """ Helper function to set requires_grad on all parameters of the model. """
    for param in module.parameters():
        param.requires_grad = requires_grad


def eval_results(pred3d, gt3d, joint_set, verbose=True, pck_threshold=150, pctiles=[99]):
    """
    Evaluates the results by printing various statistics. Also returns those results.
    Poses can be represented either in hipless 16 joints or 17 joints with hip format.
    Order is MuPo-TS order in all cases.

    Parameters:
        pred3d: dictionary of predictions in mm, seqname -> (nSample, [16|17], 3)
        gt3d: dictionary of ground truth in mm, seqname -> (nSample, [16|17], 3)
        joint_set; JointSet instance describing the order of joints
        verbose: if True, a table of the results is printed
        pctiles: list of percentiles of the errors to calculate
    Returns:
        sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles
    """

    has_hip = list(pred3d.values())[0].shape[1] == joint_set.NUM_JOINTS  # whether it contains the hip or not

    sequence_mpjpes = {}
    sequence_pcks = {}
    sequence_pctiles = {}
    all_errs = []

    for k in sorted(pred3d.keys()):
        pred = pred3d[k]
        gt = gt3d[k]

        assert pred.shape == gt.shape, "Pred shape:%s, gt shape:%s" % (pred.shape, gt.shape)
        assert (not has_hip and pred.shape[1:] == (joint_set.NUM_JOINTS - 1, 3)) or \
               (has_hip and pred.shape[1:] == (joint_set.NUM_JOINTS, 3)), \
            "Unexpected shape:" + str(pred.shape)

        errs = np.linalg.norm(pred - gt, axis=2, ord=2)  # (nSample, nJoints)

        sequence_pctiles[k] = np.nanpercentile(errs, pctiles)
        sequence_pcks[k] = np.nanmean((errs < pck_threshold).astype(np.float64))
        sequence_mpjpes[k] = np.nanmean(errs)

        # Adjusting results for missing hip
        if not has_hip:
            N = float(joint_set.NUM_JOINTS)
            sequence_pcks[k] = sequence_pcks[k] * ((N - 1) / N) + 1. / N
            sequence_mpjpes[k] = sequence_mpjpes[k] * ((N - 1) / N)

        all_errs.append(errs)

    all_errs = np.concatenate(all_errs)  # errors per joint, (nPoses, nJoints)
    joint_mpjpes = np.nanmean(all_errs, axis=0)
    joint_pctiles = np.nanpercentile(all_errs, pctiles, axis=0)

    num_joints = joint_set.NUM_JOINTS if has_hip else joint_set.NUM_JOINTS - 1
    assert_shape(all_errs, (None, num_joints))
    assert_shape(joint_mpjpes, (num_joints,))
    assert_shape(joint_pctiles, (len(pctiles), num_joints))

    if verbose:
        joint_names = joint_set.NAMES.copy()
        if not has_hip:
            joint_names = np.delete(joint_names, joint_set.index_of('hip'))  # remove root

        # Index of the percentile that will be printed. If 99 is calculated it is selected,
        # otherwise the last one
        pctile_ind = len(pctiles) - 1
        if 99 in pctiles:
            pctile_ind = pctiles.index(99)

        print("----- Per sequence and joint errors in millimeter on the validation set ----- ")
        print("%s       %6s      %5s   %6s   \t %22s  %6s     %6s" % ('Sequence', 'Avg', 'PCK', str(pctiles[pctile_ind]) + '%', '',
                                                                       'Avg', str(pctiles[pctile_ind]) + '%'))
        for seq, joint_id in zip_longest(sorted(pred3d.keys()), range(num_joints)):
            if seq is not None:
                seq_str = "%-8s:   %6.2f mm   %4.1f%%   %6.2f mm\t " \
                          % (str(seq), sequence_mpjpes[seq], sequence_pcks[seq] * 100, sequence_pctiles[seq][pctile_ind])
            else:
                seq_str = " " * 49

            if joint_id is not None:
                print('%s%15s (#%2d):  %6.2f mm   %6.2f mm ' % (seq_str, joint_names[joint_id], joint_id,
                                                                joint_mpjpes[joint_id], joint_pctiles[pctile_ind, joint_id]))
            else:
                print(seq_str)

        mean_sequence_err = np.mean(np.asarray(list(sequence_mpjpes.values()), dtype=np.float32))
        print("\nMean sequence error (Absolute MPJPE) is %6.2f mm" % mean_sequence_err)
        print("---------------------------------------------------------------- ")
        print("MRPE: %.1f" % np.mean([mrpe(pred3d[k], gt3d[k], joint_set) for k in gt3d.keys()]))

    return sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_mpjpes, joint_pctiles
