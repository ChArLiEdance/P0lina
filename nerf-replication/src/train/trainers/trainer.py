import time
import datetime
import torch
import tqdm
from torch.nn.parallel import DistributedDataParallel, DataParallel
from src.config import cfg
from src.utils.data_utils import to_cuda


class Trainer(object):
    def __init__(self, network):
        # 检测GPU数量和配置
        self.num_gpus = torch.cuda.device_count()
        self.use_multi_gpu = self.num_gpus > 1 and len(cfg.gpus) > 1
        
        # 设置设备
        if self.use_multi_gpu:
            # 多GPU模式：使用DataParallel（更简单，适合单机多卡）
            device = torch.device("cuda:0")  # DataParallel总是使用cuda:0
            network = network.to(device)
            
            # 使用DataParallel包装网络
            network = DataParallel(
                network, 
                device_ids=cfg.gpus,
                output_device=0
            )
            
            # 调整配置以优化多GPU性能
            if hasattr(cfg.task_arg, 'chunk_size'):
                # 根据GPU数量调整chunk_size，避免内存不足
                original_chunk = cfg.task_arg.chunk_size
                cfg.task_arg.chunk_size = max(original_chunk // self.num_gpus, 1024)
                print(f"多GPU优化: chunk_size从{original_chunk}调整为{cfg.task_arg.chunk_size}")
            
            if hasattr(cfg.task_arg, 'N_rays'):
                # 根据GPU数量调整N_rays，提高并行效率
                original_rays = cfg.task_arg.N_rays
                cfg.task_arg.N_rays = original_rays * self.num_gpus
                print(f"多GPU优化: N_rays从{original_rays}调整为{cfg.task_arg.N_rays}")
                
        else:
            # 单GPU或分布式模式
            device = torch.device("cuda:{}".format(cfg.local_rank))
            network = network.to(device)
            if cfg.distributed:
                network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
                network = DistributedDataParallel(
                    network,
                    device_ids=[cfg.local_rank],
                    output_device=cfg.local_rank,
                    find_unused_parameters=True,
                )
        
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device
        self.global_step = 0
        self.use_multi_gpu = self.use_multi_gpu
        
        # 显示GPU信息
        if cfg.local_rank == 0 or not cfg.distributed:
            print(f"=== GPU配置信息 ===")
            print(f"检测到 {self.num_gpus} 个GPU")
            print(f"配置使用GPU: {cfg.gpus}")
            print(f"并行模式: {'DataParallel' if self.use_multi_gpu else 'Single GPU' if self.num_gpus == 1 else 'DistributedDataParallel'}")
            
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU #{i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if self.use_multi_gpu:
                print(f"多GPU优化已启用:")
                print(f"  - chunk_size: {cfg.task_arg.chunk_size}")
                print(f"  - N_rays: {cfg.task_arg.N_rays}")
                #print(f"  - 预期加速比: ~{self.num_gpus}x")

    def reduce_loss_stats(self, loss_stats):
        if self.use_multi_gpu:
            # DataParallel会自动处理loss reduction
            reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        else:
            reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        
        # 创建进度条 - 简化显示
        if cfg.local_rank == 0 or not cfg.distributed:
            pbar = tqdm.tqdm(total=max_iter, desc=f"Epoch {epoch}", 
                           leave=True, ncols=80, position=0)
        
        # 性能监控
        total_batch_time = 0
        total_data_time = 0
        
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = to_cuda(batch, self.device)
            batch["step"] = self.global_step
            
            # 记录前向传播开始时间
            forward_start = time.time()
            
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()
            
            # 记录总batch时间
            batch_time = time.time() - end
            total_batch_time += batch_time
            total_data_time += data_time

            if cfg.local_rank > 0 and cfg.distributed:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            self.global_step += 1
            
            # 只更新进度条，不显示详细信息
            if cfg.local_rank == 0 or not cfg.distributed:
                pbar.update(1)
            
            # 每log_interval步在进度条下方显示详细信息
            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                if cfg.local_rank == 0 or not cfg.distributed:
                    eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    lr = optimizer.param_groups[0]["lr"]
                    memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    fps = 1.0 / batch_time if batch_time > 0 else 0
                    
                    # 计算平均性能指标
                    avg_batch_time = total_batch_time / iteration
                    avg_data_time = total_data_time / iteration
                    
                    # 显示性能信息
                    perf_info = ""
                    if self.use_multi_gpu:
                        # single_gpu_time = avg_batch_time * self.num_gpus
                        # speedup = single_gpu_time / avg_batch_time
                        # perf_info = f"  speedup: {speedup:.2f}x"
                    
                    print(f"eta: {eta_string}  epoch: {epoch}  step: {iteration}  loss: {loss.item():.4f}  "
                          f"data: {avg_data_time:.4f}  batch: {avg_batch_time:.4f}{perf_info}  "
                          f"lr: {lr:.6f}  max_mem: {memory:.0f}MB")

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record("train")
        
        # 关闭进度条
        if cfg.local_rank == 0 or not cfg.distributed:
            pbar.close()
            
            # 显示epoch总结
            if self.use_multi_gpu:
                print(f"Epoch {epoch} 完成 - 平均batch时间: {total_batch_time/max_iter:.4f}s")

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        image_stats = {}
        data_size = len(data_loader)
        
        # 创建验证进度条 - 简化显示
        if cfg.local_rank == 0 or not cfg.distributed:
            pbar = tqdm.tqdm(total=data_size, desc=f"Validation Epoch {epoch}", 
                           leave=True, ncols=80, position=0)
        
        for batch in data_loader:
            batch = to_cuda(batch, self.device)
            batch["step"] = recorder.step
            with torch.no_grad():
                output, loss, loss_stats, _ = self.network(batch)
                if evaluator is not None:
                    image_stats_ = evaluator.evaluate(output, batch)
                    if image_stats_ is not None:
                        image_stats.update(image_stats_)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v
            
            # 只更新进度条，不显示详细信息
            if cfg.local_rank == 0 or not cfg.distributed:
                pbar.update(1)

        # 关闭验证进度条
        if cfg.local_rank == 0 or not cfg.distributed:
            pbar.close()

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append("{}: {:.4f}".format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record("val", epoch, val_loss_stats, image_stats)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.tensor(10., device=x.device))
