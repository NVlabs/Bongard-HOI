# ----------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Bongard-HOI. To view a copy of this license, see the LICENSE file.
# ----------------------------------------------------------------------

import argparse
import os
from torch.utils import data
import yaml
import numpy as np
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from tqdm import tqdm

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.image_bongard_bbox import collate_images_boxes_dict

from detectron2.structures import Boxes


def main(config):
    args.gpu = ''#[i for i in range(torch.cuda.device_count())]
    args.train_gpu = [i for i in range(torch.cuda.device_count())]
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus - 1):
        args.gpu += '{},'.format(i)
    args.gpu += '{}'.format(num_gpus - 1)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu
    utils.set_gpu(args.gpu)
    args.config = config

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.sync_bn = True
        port = utils.find_free_port()
        args.dist_url = args.dist_url.format(port)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    config = args.config
    svname = args.name
    if svname is None:
        config_name, _ = os.path.splitext(os.path.basename(args.config_file))
        svname = '{}shot'.format(config['n_shot'])
        svname += '_' + config['model']
        if config['model_args'].get('encoder'):
            svname += '-' + config['model_args']['encoder']
        svname = os.path.join(config_name, config['train_dataset'], svname)
    if not args.test_only:
        svname += '-seed' + str(args.seed)
    if args.tag is not None:
        svname += '_' + args.tag

    sub_dir_name = 'default'
    if args.opts:
        sub_dir_name = args.opts[0]
        split = '#'
        for opt in args.opts[1:]:
            sub_dir_name += split + opt
            split = '#' if split == '_' else '_'
    svname = os.path.join(svname, sub_dir_name)

    if utils.is_main_process() and not args.test_only:
        save_path = os.path.join(args.save_dir, svname)
        utils.ensure_path(save_path, remove=False)
        utils.set_log_path(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        args.writer = writer

        yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

        logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)
    else:
        save_path = None
        writer = None
        args.writer = writer
        logger = None

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    args.n_train_way = n_train_way
    args.n_train_shot = n_train_shot
    args.n_query = n_query
    args.n_shot = n_shot
    args.n_way = n_way

    # train
    dataset_configs = config['train_dataset_args']
    dataset_configs['use_gt_bbox'] = config['use_gt_bbox']
    train_dataset = datasets.make(config['train_dataset'], **dataset_configs)
    if utils.is_main_process():
        utils.log('train dataset: {} samples'.format(len(train_dataset)))
    if args.distributed:
        args.batch_size = int(ep_per_batch / ngpus_per_node)
        args.batch_size_val = int(ep_per_batch / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    else:
        args.batch_size = ep_per_batch
        args.batch_size_val = ep_per_batch
        args.workers = args.workers

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=False,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_images_boxes_dict
    )

    # val & test
    if args.test_only:
        val_type_list = [
            'test_seen_obj_seen_act',
            'test_seen_obj_unseen_act',
            'test_unseen_obj_seen_act',
            'test_unseen_obj_unseen_act'
        ]
    else:
        val_type_list = [
            'val_seen_obj_seen_act',
            'val_seen_obj_unseen_act',
            'val_unseen_obj_seen_act',
            'val_unseen_obj_unseen_act'
        ]

    val_loader_dict = {}
    for val_type_i in val_type_list:
        dataset_configs = config['{}_dataset_args'.format(val_type_i)]
        dataset_configs['use_gt_bbox'] = config['use_gt_bbox']
        val_dataset_i = datasets.make(config['{}_dataset'.format(val_type_i)], **dataset_configs)
        if utils.is_main_process():
            utils.log('{} dataset: {} samples'.format(val_type_i, len(val_dataset_i)))

        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_i)
        else:
            val_sampler = None
        val_loader_i = torch.utils.data.DataLoader(
            val_dataset_i,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=collate_images_boxes_dict
        )
        val_loader_dict[val_type_i] = val_loader_i

    ########

    #### Model and optimizer ####

    if config.get('load'):
        print('loading pretrained model: ', config['load'])
        model = models.load(torch.load(config['load']))
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            print('loading pretrained encoder: ', config['load_encoder'])
            pretrain = config.get('encoder_pretrain').lower()
            if pretrain != 'scratch':
                pretrain_model_path = config['load_encoder'].format(pretrain)
                state_dict = torch.load(pretrain_model_path, map_location='cpu')
                missing_keys, unexpected_keys = model.encoder.encoder.load_state_dict(state_dict, strict=False)
                for key in missing_keys:
                    assert key.startswith('g_mlp.') \
                        or key.startswith('proj') \
                        or key.startswith('trans') \
                        or key.startswith('roi_processor') \
                        or key.startswith('roi_dim_processor') \
                        or key.startswith('classifier'), key
                for key in unexpected_keys:
                    assert key.startswith('fc.')
                if utils.is_main_process():
                    utils.log('==> Successfully loaded {} for the enocder.'.format(pretrain_model_path))

        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if utils.is_main_process():
            utils.log(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    if utils.is_main_process() and not args.test_only:
        utils.log('num params: {}'.format(utils.compute_n_params(model)))
        utils.log('Results will be saved to {}'.format(save_path))

    max_steps = min(len(train_loader), config['train_batches']) * config['max_epoch']
    optimizer, lr_scheduler, update_lr_every_epoch = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], max_steps, **config['optimizer_args']
    )
    assert lr_scheduler is not None
    args.update_lr_every_epoch = update_lr_every_epoch

    if args.test_only:
        filename = args.test_model
        assert os.path.exists(filename)
        ckpt = torch.load(filename, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if utils.is_main_process():
            utils.log('==> Sucessfully resumed from a checkpoint {}'.format(filename))
    else:
        start_epoch = 0

    ######## Training & Validation

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    if args.test_only:
        test_acc = []
        test_acc_type = []
        for val_type_i, val_loader_i in val_loader_dict.items():
            loss_val_i, acc_val_i = validate(val_loader_i, model, 0)
            test_acc.append(acc_val_i)
            test_acc_type.append(val_type_i)
            if utils.is_main_process():
                print('testing result: ', val_type_i, acc_val_i)
        print('summary')
        for i, j in zip(test_acc_type, test_acc):
            print(i, j)
        print('avg', sum(test_acc)/4)
        return 0

    best_val_result = {type: 0.0 for type in val_loader_dict.keys()}
    for epoch in range(start_epoch, max_epoch):

        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, acc_train = train(train_loader, model, optimizer, lr_scheduler, epoch_log, args)
        if args.update_lr_every_epoch:
            lr_scheduler.step()
        if utils.is_main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('acc_train', acc_train, epoch_log)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param)

        if epoch_log % config['eval_epoch'] == 0:
            avg_acc_val = 0
            for val_type_i, val_loader_i in val_loader_dict.items():
                loss_val_i, acc_val_i = validate(val_loader_i, model, epoch_log)
                if acc_val_i > best_val_result[val_type_i]:
                    best_val_result[val_type_i] = acc_val_i
                if utils.is_main_process():
                    utils.log('{} result: loss {:.4f}, acc: {:.4f}.'.format(val_type_i, loss_val_i, acc_val_i))
                    writer.add_scalar('loss_{}'.format(val_type_i), loss_val_i, epoch_log)
                    writer.add_scalar('acc_{}'.format(val_type_i), acc_val_i, epoch_log)
                avg_acc_val += acc_val_i
            avg_acc_val /= len(val_loader_dict.keys())

        utils.log('Best val results so far:')
        utils.log(best_val_result)

        if avg_acc_val > max_va and utils.is_main_process():
            max_va = avg_acc_val
            filename = os.path.join(save_path, 'best_model.pth')
            ckpt = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(ckpt, filename)
        if utils.is_main_process():
            writer.flush()

    if utils.is_main_process():
        logger.close()

def train(train_loader, model, optimizer, lr_scheduler, epoch, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    main_loss_meter = utils.AverageMeter()
    aux_loss_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    intersection_meter = utils.AverageMeter()
    union_meter = utils.AverageMeter()
    target_meter = utils.AverageMeter()

    config = args.config

    # train
    model.train()

    if utils.is_main_process():
        args.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    lr = optimizer.param_groups[0]['lr']

    end = time.time()
    max_iter = config['max_epoch'] * len(train_loader)
    for batch_idx, data_dict in enumerate(train_loader):
        if batch_idx >= config['train_batches']:
            break

        x_shot = data_dict['shot_ims'].cuda(non_blocking=True)
        x_query = data_dict['query_ims'].cuda(non_blocking=True)
        label_query = data_dict['query_labs'].cuda(non_blocking=True).view(-1)
        if 'shot_boxes' in data_dict:
            assert 'query_boxes' in data_dict
            assert 'shot_boxes_dim' in data_dict
            assert 'query_boxes_dim' in data_dict
            shot_boxes = data_dict['shot_boxes']
            for idx, shot_boxes_i in enumerate(shot_boxes):
                shot_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_boxes_i]
                shot_boxes[idx] = shot_boxes_i

            query_boxes = data_dict['query_boxes']
            for idx, query_boxes_i in enumerate(query_boxes):
                query_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_boxes_i]
                query_boxes[idx] = query_boxes_i

            shot_boxes_dim = data_dict['shot_boxes_dim'].cuda(non_blocking=True)
            query_boxes_dim = data_dict['query_boxes_dim'].cuda(non_blocking=True)
        else:
            shot_boxes = None
            query_boxes = None
            shot_boxes_dim = None
            query_boxes_dim = None

        data_time.update(time.time() - end)

        if args.config['model'] == 'snail':  # only use one selected label_query
            query_dix = np.random.randint(args.n_train_way * args.n_query)
            label_query = label_query.view(args.batch_size, -1)[:, query_dix]
            x_query = x_query[:, query_dix: query_dix + 1]
            if query_boxes is not None:
                for ii, boxes_list_i in enumerate(query_boxes):
                    assert len(boxes_list_i) == args.n_train_way * args.n_query
                    query_boxes[ii] = boxes_list_i[query_dix:query_dix+1]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if shot_boxes is not None and query_boxes is not None:
                logits = model(
                    x_shot,
                    x_query,
                    shot_boxes=shot_boxes,
                    query_boxes=query_boxes,
                    shot_boxes_dim=shot_boxes_dim,
                    query_boxes_dim = query_boxes_dim
                ).view(-1, args.n_train_way)
            else:
                logits = model(x_shot, x_query).view(-1, args.n_train_way)
            loss = F.cross_entropy(logits, label_query)
            acc = utils.compute_acc(logits, label_query)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lrs = lr_scheduler.get_last_lr()
        if not args.update_lr_every_epoch:
            lr_scheduler.step()

        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            acc = acc * n
            count = label_query.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(acc)
            dist.all_reduce(count)
            n = count.item()
            loss = loss / n
            acc = acc / n

        loss_meter.update(loss.item(), logits.size(0))
        acc_meter.update(acc.item(), logits.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        current_iter = epoch * len(train_loader) + batch_idx + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (batch_idx + 1) % config['print_freq'] == 0 and utils.is_main_process():
            utils.log(
                'Epoch: [{}/{}][{}/{}] '
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Remain {remain_time} '
                'Loss {loss_meter.val:.4f} '
                'Acc {acc_meter.val:.4f} '
                'lr {lr:.6f}'.format(
                    epoch, config['max_epoch'], batch_idx + 1, len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    remain_time=remain_time,
                    loss_meter=loss_meter,
                    acc_meter=acc_meter,
                    lr=lrs[0]
                )
            )

    if utils.is_main_process():
        utils.log('Train result at epoch [{}/{}]: loss {:.4f}, acc {:.4f}.'.format(epoch, config['max_epoch'], loss_meter.avg, acc_meter.avg))
    return loss_meter.avg, acc_meter.avg

def validate(val_loader, model, epoch_log):
    # eval
    model.eval()

    config = args.config
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()

    np.random.seed(0)
    for data_dict in tqdm(val_loader):
        x_shot = data_dict['shot_ims'].cuda(non_blocking=True)
        x_query = data_dict['query_ims'].cuda(non_blocking=True)
        label_query = data_dict['query_labs'].cuda(non_blocking=True).view(-1)
        if 'shot_boxes' in data_dict:
            assert 'query_boxes' in data_dict
            shot_boxes = data_dict['shot_boxes']
            for idx, shot_boxes_i in enumerate(shot_boxes):
                shot_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in shot_boxes_i]
                shot_boxes[idx] = shot_boxes_i

            query_boxes = data_dict['query_boxes']
            for idx, query_boxes_i in enumerate(query_boxes):
                query_boxes_i = [Boxes(boxes.tensor.cuda(non_blocking=True)) for boxes in query_boxes_i]
                query_boxes[idx] = query_boxes_i

            assert 'shot_boxes_dim' in data_dict
            assert 'query_boxes_dim' in data_dict
            shot_boxes_dim = data_dict['shot_boxes_dim'].cuda(non_blocking=True)
            query_boxes_dim = data_dict['query_boxes_dim'].cuda(non_blocking=True)
        else:
            shot_boxes = None
            query_boxes = None
            shot_boxes_dim = None
            query_boxes_dim = None

        if config['model'] == 'snail':  # only use one randomly selected label_query
            query_dix = np.random.randint(args.n_train_way)
            label_query = label_query.view(-1, args.n_train_way * args.n_query)[:, query_dix]
            x_query = x_query[:, query_dix: query_dix + 1]
            if query_boxes is not None:
                for ii, boxes_list_i in enumerate(query_boxes):
                    assert len(boxes_list_i) == args.n_train_way * args.n_query
                    query_boxes[ii] = boxes_list_i[query_dix:query_dix+1]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                if shot_boxes is not None and query_boxes is not None:
                    logits = model(
                        x_shot,
                        x_query,
                        shot_boxes=shot_boxes,
                        query_boxes=query_boxes,
                        shot_boxes_dim=shot_boxes_dim,
                        query_boxes_dim=query_boxes_dim,
                        eval=True
                    ).view(-1, args.n_train_way)
                else:
                    logits = model(x_shot, x_query, eval=True).view(-1, args.n_train_way)
                loss = F.cross_entropy(logits, label_query)
                acc = utils.compute_acc(logits, label_query)

        n = logits.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            acc = acc * n
            count = logits.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(acc)
            dist.all_reduce(count)
            n = count.item()
            loss = loss / n
            acc = acc / n
        else:
            loss = torch.mean(loss)
            acc = torch.mean(acc)

        loss_meter.update(loss.item(), logits.size(0))
        acc_meter.update(acc.item(), logits.size(0))
    return loss_meter.avg, acc_meter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    parser.add_argument('--name', default=None)
    parser.add_argument('--save_dir', default='./save_dist')
    parser.add_argument('--tag', default=None)
    # parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--test_model', default=None)

    # distributed training
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-backend', default='nccl')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.multiprocessing_distributed = True

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    if args.opts is not None:
        config = utils.override_cfg_from_list(config, args.opts)
    print('config:')
    print(config)
    main(config)
