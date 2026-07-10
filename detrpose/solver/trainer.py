"""
DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM (https://github.com/Intellindust-AI-Lab/DEIM/)
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/)
Copyright (c) 2023 RT-DETR Authors. All Rights Reserved.
"""

import json
import logging
import datetime
from pathlib import Path
from omegaconf import OmegaConf
from collections import Counter


import time
import atexit
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.metrics import BestMetricHolder
from src.utils.profiler import stats
from src.utils import dist as dist_utils
from src.core import instantiate

from .engine import train_one_epoch, evaluate

def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def _setup(self,):
        """Avoid instantiating unnecessary classes"""
        dist_utils.init_distributed_mode(self.cfg.public)
        args = self.cfg.public

        # fix the seed for reproducibility
        seed = args.seed + dist_utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_without_ddp = instantiate(self.cfg.model).to(self.device)
        self.model = dist_utils.warp_model(
            self.model_without_ddp.to(args.device), 
            sync_bn=args.sync_bn, 
            find_unused_parameters=args.find_unused_params
            )

        self.postprocessor = instantiate(self.cfg.postprocessor)
        self.evaluator = instantiate(self.cfg.evaluator)

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.cfg[name])
        global_cfg = self.cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        num_gpus = dist_utils.get_world_size()
        print(f'building {name} with batch_size={bs} and {num_gpus} GPUs ...')
        dataloader = self.cfg[name]
        dataloader.batch_size = bs
        loader = instantiate(dataloader)
        loader.shuffle = dataloader.get('shuffle', False)
        return loader

    def evaluation(self, ):
        self._setup()
        args = self.cfg.public
        self.args = args

        if self.cfg.public.use_ema:
            self.cfg.ema.model = self.model_without_ddp
            self.ema = instantiate(self.cfg.ema)
        else:
            self.ema = None

        # Load datasets
        if args.eval:
            dataset_val = self.build_dataloader('dataset_val')
        else:
            dataset_val = self.build_dataloader('dataset_test')

        self.dataloader_val = dist_utils.warp_loader(dataset_val, self.cfg.dataset_val.shuffle)

        if hasattr(args, 'resume'):
            self.resume()
        else:
            raise "Use resume during evaluation"

    def train(self,):
        self._setup()
        args = self.cfg.public
        self.args = args

        self.writer = SummaryWriter(self.output_dir/"summary")
        atexit.register(self.writer.close)

        if dist_utils.is_main_process():
            self.writer.add_text("config", "{:s}".format(OmegaConf.to_yaml(self.cfg).__repr__()), 0)
        
        if self.cfg.public.use_ema:
            self.cfg.ema.model = self.model_without_ddp
            self.ema = instantiate(self.cfg.ema)
        else:
            self.ema = None

        self.criterion = instantiate(self.cfg.criterion)

        self.cfg.optimizer.params.model = self.model_without_ddp
        self.optimizer = instantiate(self.cfg.optimizer)

        self.cfg.lr_scheduler.optimizer = self.optimizer
        self.lr_scheduler = instantiate(self.cfg.lr_scheduler)

        if hasattr(self.cfg, 'warmup_scheduler'):
            self.cfg.warmup_scheduler.lr_scheduler = self.lr_scheduler
            self.warmup_scheduler = instantiate(self.cfg.warmup_scheduler)
        else:
            self.warmup_scheduler = None

        # Load datasets
        dataset_train = self.build_dataloader('dataset_train')
        dataset_val = self.build_dataloader('dataset_val')

        self.dataloader_train = dist_utils.warp_loader(dataset_train, self.cfg.dataset_train.shuffle)
        self.dataloader_val = dist_utils.warp_loader(dataset_val, self.cfg.dataset_val.shuffle)

        assert not (hasattr(args, 'resume') and hasattr(args, 'pretrain')) #'You cant resume and pretain at the same time. Choose one.' 
        if hasattr(args, 'resume'):
            self.resume()

        if hasattr(args, 'pretrain'):
            self.pretrain(args.pretrain)

        self.best_map_holder = BestMetricHolder(use_ema=args.use_ema)

    def fit(self,):
        self.train()
        args = self.args
        model_stats = stats(self.model_without_ddp)
        print(model_stats)
        
        print("-" * 42 + "Start training" + "-" * 43)
        
        if hasattr(args, 'resume'):
            module = self.ema.module if self.ema is not None else self.model
            test_stats = evaluate(
                module, 
                self.postprocessor, 
                self.evaluator,
                self.dataloader_val, 
                self.device, 
                self.writer
            )

            map_regular = test_stats["coco_eval_keypoints"][0]
            _isbest = self.best_map_holder.update(map_regular, args.start_epoch-1, is_ema=False)

        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()

            self.dataloader_train.set_epoch(epoch)
            # self.dataloader_train.dataset.set_epoch(epoch)
            if dist_utils.is_dist_avail_and_initialized():
                self.dataloader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.dataloader_train, 
                self.optimizer, 
                self.cfg.dataset_train.batch_size,
                args.grad_accum_steps,
                self.device, 
                epoch,
                args.clip_max_norm, 
                lr_scheduler=self.lr_scheduler, 
                warmup_scheduler=self.warmup_scheduler, 
                writer=self.writer, 
                args=args,
                ema=self.ema
                )

            if self.warmup_scheduler is None or self.warmup_scheduler.finished():
                self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.save_checkpoint_interval == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': self.model_without_ddp.state_dict(),
                        'ema': self.ema.state_dict() if self.ema is not None else None,
                        'optimizer': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict(),
                        'warmup_scheduler': self.warmup_scheduler.state_dict() if self.warmup_scheduler is not None else None,
                        'epoch': epoch,
                        'args': args,
                    }
                    dist_utils.save_on_master(weights, checkpoint_path)

            module = self.ema.module if self.ema is not None else self.model
                    
            # eval
            test_stats = evaluate(
                module, 
                self.postprocessor, 
                self.evaluator,
                self.dataloader_val, 
                self.device, 
                self.writer
            )

            if self.writer is not None and dist_utils.is_main_process():
                coco_stats = test_stats['coco_eval_keypoints']
                coco_names = ["sAP50:95", "sAP50", "sAP75", "sAP50:95-Medium", "sAP50:95-Large"]
                for k, val in zip(coco_names, coco_stats):
                    self.writer.add_scalar(f"Test/{k}", val, epoch)
                
            log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': model_stats['params']
                }

            map_regular = test_stats["coco_eval_keypoints"][0]
            _isbest = self.best_map_holder.update(map_regular, epoch, is_ema=False)

            if _isbest:
                print(f"New best achieved @ epoch {epoch:04d}!!!...")
                checkpoint_path = self.output_dir / 'checkpoint_best_regular.pth'
                weights = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema': self.ema.state_dict() if self.ema is not None else None,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'warmup_scheduler': self.warmup_scheduler.state_dict() if self.warmup_scheduler is not None else None,
                    'epoch': epoch,
                    'args': args,
                }
                dist_utils.save_on_master(weights, checkpoint_path)

            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if self.evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "keypoints" in self.evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(self.evaluator.coco_eval["keypoints"].eval,
                                       self.output_dir / "eval" / name)
        self.writer.close()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))    

    def eval(self, ):
        self.evaluation()
        module = self.ema.module if self.ema is not None else self.model

        # eval
        test_stats = evaluate(
            module, 
            self.postprocessor, 
            self.evaluator,
            self.dataloader_val, 
            self.device, 
        )

    def test(self, ):
        self.evaluation()
        module = self.ema.module if self.ema is not None else self.model

        # eval
        res_json = evaluate(
            module, 
            self.postprocessor, 
            None, #self.evaluator,
            self.dataloader_val, 
            self.device, 
            save_results=True
        )

        print("Saving results in results.json ...")
        with open("results.json", "w") as final:
            json.dump(res_json, final)
        print("Done ...")

    def resume(self,):
        args = self.cfg.public
        if hasattr(args, "resume") and len(args.resume)>0:
            print(f"Loading weights from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            self.model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            if self.ema:
                if 'ema' in checkpoint: 
                    self.ema.load_state_dict(checkpoint['ema'], strict=False)
                else:
                    self.ema.module.load_state_dict(checkpoint['model'], strict=False)

            if not(args.eval or args.test) and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                import copy
                p_groups = copy.deepcopy(self.optimizer.param_groups)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(self.optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
                if self.warmup_scheduler:
                    self.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler'])

                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                args.override_resumed_lr_drop = True
                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    self.lr_scheduler.milestones = Counter(self.cfg.lr_scheduler.milestones)
                    self.lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))
                self.lr_scheduler.step(self.lr_scheduler.last_epoch)
                args.start_epoch = checkpoint['epoch'] + 1
        else:
            print("Initializing the model with random parameters!")


    def pretrain(self, model_name):
        arch_configs = {
            # COCO
            'dfine_n_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth',
            'dfine_s_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth',
            'dfine_m_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth',
            'dfine_l_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth',
            'dfine_x_coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth',
            # OBJECT 365
            'dfine_s_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj365.pth',
            'dfine_m_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj365.pth',
            'dfine_l_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj365.pth',
            'dfine_x_obj365': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj365.pth',
            # OBJECT 365 + COCO
            'dfine_s_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_obj2coco.pth',
            'dfine_m_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth',
            'dfine_l_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco.pth',
            'dfine_x_obj2coco': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth',
        }
        RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"

        download_url = arch_configs[model_name]
        local_model_dir = './weight/dfine/'

        try:
            # If the file doesn't exist locally, download from the URL
            if safe_get_rank() == 0:
                print(
                    GREEN
                    + "If the pretrained D-FINE can't be downloaded automatically. Please check your network connection."
                    + RESET
                )
                print(
                    GREEN
                    + "Please check your network connection. Or download the model manually from "
                    + RESET
                    + f"{download_url}"
                    + GREEN
                    + " to "
                    + RESET
                    + f"{local_model_dir}."
                    + RESET
                )
                state = torch.hub.load_state_dict_from_url(
                    download_url, map_location="cpu", model_dir=local_model_dir
                )
                print(f"Loaded pretrained DFINE from URL.")

            # Wait for rank 0 to download the model
            safe_barrier()

            # All processes load the downloaded model
            model_path = local_model_dir  + model_name + ".pth"

            state = torch.load(model_path, map_location="cpu", weights_only=False)

            if "ema" in state:
                print("USING EMA WEIGHTS!!!")
                pretrain_state_dict = state["ema"]["module"]
            else:
                pretrain_state_dict = state["model"]
            
            new_state_dict = {}
            for k in pretrain_state_dict:
                if ("decoder" in k):
                    continue
                new_state_dict[k] = state['model'][k]

            print(f"⚠️  Loading weights for the backbone and decoder from {model_name} ⚠️")
            missing_keys, unexpected_keys = self.model_without_ddp.load_state_dict(new_state_dict, strict=False)

            if len(unexpected_keys) > 0:
                print("Warning. The following RTGroupPose does not have the following parameters:")
                for k in unexpected_keys:
                    print(f"    - {k}")
            else:
                print(f'✅ Successfully initilized the backbone and encoder using {model_name} weights ✅')

            missing_keys, unexpected_keys = self.ema.module.load_state_dict(new_state_dict, strict=False)

        except (Exception, KeyboardInterrupt) as e:
            if safe_get_rank() == 0:
                print(f"{str(e)}")
                logging.error(
                    RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET
                )
                logging.error(
                    GREEN
                    + "Please check your network connection. Or download the model manually from "
                    + RESET
                    + f"{download_url}"
                    + GREEN
                    + " to "
                    + RESET
                    + f"{local_model_dir}."
                    + RESET
                )
            exit()
