import torch
import torch.multiprocessing as mp
from trainer import Trainer


def distributed_main(args):
    ngpus_per_node = len(args.devices)
    assert ngpus_per_node <= torch.cuda.device_count(), "GPU device num exceeds max capacity."

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function

    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def worker(gpu, ngpus_per_node, args):
    trainer = Trainer(args)

    trainer.make_model_env(gpu, ngpus_per_node)
    # model.make_run_env()
    trainer.train()
