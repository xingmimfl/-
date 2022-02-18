import argparse
import os
import random
import shutil
import time
import warnings

import torch

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

if __name__=="__main__":
    #----解析torch.distributed.launch的参数
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    #----启动分布式训练---
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)
    torch.distributed.barrier()

    
    the_device = torch.device("cuda:%d" % local_rank)

    #----模型做分布式
    model = create_model()
    model.to(the_device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    #----loss函数
    criterion = nn.CrossEntropyLoss().to(the_device)

    #----optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #-----loading training dataset----
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    #-----training----
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch) #----不可缺少的一步
        #val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args) #---调整学习率

        #----one iteration in training----
        for i, (images, target) in enumerate(train_loader):
            images = images.to(the_device)
            target = target.to(the_device)

            output = model(images)
            loss = criterion(output, target)
            torch.distributed.barrier() #----同步

            reduced_loss = reduce_mean(loss, args.nprocs)
            losses.update(reduced_loss.item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #----one iteration in training----

        if args.local_rank == 0: #---只有local_rank=0的process保存模型
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

