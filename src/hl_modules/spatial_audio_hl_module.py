import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch
from numpy import mean
from src.metrics.metrics import Metrics
from src.metrics.metrics import compute_decay
import src.utils as utils

class FakeModel(nn.Module):
    def __init__(self, model):
        super(FakeModel, self).__init__()
        self.model = model

class PLModule(object):
    def __init__(self, model, model_params, sr,
                 optimizer, optimizer_params,
                 scheduler=None, scheduler_params=None,
                 loss=None, loss_params=None, 
                 metrics=[], init_ckpt=None,
                 grad_clip=None,
                 use_dp=True,
                 samples_per_speaker_number=3):
        
        self.model = model(**model_params)
        self.use_dp = use_dp
        if use_dp:
            self.model = nn.DataParallel(self.model)
        
        self.sr = sr
        self.samples_per_speaker_number = samples_per_speaker_number

        self.monitor = 'val/loss'
        self.monitor_mode = 'min'
        self.mode = None
        self.val_samples = {}
        self.train_samples = {}
        self.loss_fn = loss(**loss_params)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)
        self.grad_clip = grad_clip
        self.metric= Metrics("mse")
        self.loss_fn = utils.import_attr(loss)(**loss_params)
        if self.grad_clip is not None:
            print(f"USING GRAD CLIP: {self.grad_clip}")
        else:
            print("ERROR! NOT USING GRAD CLIP" * 100)

        self.scheduler = self.init_scheduler(scheduler, scheduler_params)
        self.scheduler_params = scheduler_params
        self.epoch = 0
        if init_ckpt is not None:
        if init_ckpt.endswith('.ckpt'):
            state = torch.load(init_ckpt)['state_dict']
            # print(state.keys())
            
            if self.use_dp:
                _model = self.model.module
            else:
                _model = self.model
            
            mdl = FakeModel(_model)
            mdl.load_state_dict(state)
            self.model = nn.DataParallel(mdl.model)
        else:
            state = torch.load(init_ckpt)['model']
            
            if self.use_dp:
                self.model.module.load_state_dict(state)
            else:
                self.model.load_state_dict(state)
    def load_state(self, path, map_location=None):
        breakpoint()
        state = torch.load(path, map_location=map_location)

        if self.use_dp:
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])
        
        self.optimizer = optim.Adam(self.model.parameters(), **self.scheduler_params)
        if self.scheduler is not None:
            self.scheduler = self.init_scheduler(self.scheduler_name, self.scheduler_params)

        self.optimizer.load_state_dict(state['optimizer'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        
        self.epoch = state['current_epoch']
        self.metric_values = state['metric_values']


    def dump_state(self, path):
        if self.use_dp:
            _model = self.model.module
        else:
            _model = self.model
        
        state = dict(model = _model.state_dict(),
                     optimizer = self.optimizer.state_dict(),
                     current_epoch = self.epoch,
                     metric_values=self.metric_values
                     )
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        
        torch.save(state, path)

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        print()
        print("=" * 25, "STARTING EPOCH", self.epoch, "=" * 25)
        print()
    

    def get_avg_metric_at_epoch(self, metric, epoch=None):
        if epoch is None:
            epoch = self.epoch
        
        return self.metric_values[epoch][metric]['epoch'] / \
            self.metric_values[epoch][metric]['num_elements']

    def on_epoch_end(self, best_path):
        assert self.epoch + 1 == len(self.metric_values), \
            "Current epoch must be equal to length of metrics (0-indexed)"

        monitor_metric_last = self.get_avg_metric_at_epoch(self.monitor)

        # Go over all epochs
        # Don't save if metric is nan for whatever reason
        # save = True
        save = monitor_metric_last is not torch.nan
        for epoch in range(len(self.metric_values) - 1):
            monitor_metric_at_epoch = self.get_avg_metric_at_epoch(self.monitor, epoch)
            
            if self.monitor_mode == 'max':
                # If there is any model with monitor larger than current, then
                # this is not the best model
                if monitor_metric_last < monitor_metric_at_epoch:
                    save = False
                    break

            if self.monitor_mode == 'min':
                # If there is any model with monitor smaller than current, then
                # this is not the best model
                if monitor_metric_last > monitor_metric_at_epoch:
                    save = False
                    break
        
        # If this is best, save it
        if save:
            print("Current checkpoint is the best! Saving it...")
            self.dump_state(best_path)
        
        val_loss = self.get_avg_metric_at_epoch('val/loss')
        # val_snr_i = self.get_avg_metric_at_epoch('val/snr_i') #commented out because we dont have snr
        # val_si_sdr_i = self.get_avg_metric_at_epoch('val/si_sdr_i')

        print(f'Val loss: {val_loss:.02f}')
   
        self.train_samples.clear()

        # for spk_num in self.val_samples:
        #     log_audio(wandb_run, f"val/audio_samples_{spk_num}spk", self.val_samples[spk_num], sr=self.sr)
        self.val_samples.clear()

        # wandb_run.log({'epoch': self.epoch}, commit=True, step=self.epoch + 1)
        
        if self.scheduler is not None:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # Get last metric
                self.scheduler.step(monitor_metric_last)
            else:
                self.scheduler.step()

        self.epoch += 1


    def log_metric(self, name, value, batch_size=1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True):
        """
        Logs a metric
        value must be the AVERAGE value across the batch
        Must provide batch size for accurate average computation
        """
        
        epoch_str = self.epoch
        if epoch_str not in self.metric_values:
            self.metric_values[epoch_str] = {}

        if (name not in self.metric_values[epoch_str]):
            self.metric_values[epoch_str][name] = dict(step=None, epoch=None)
        
        if type(value) == torch.Tensor:
            value = value.item()

        if on_step:            
            if self.metric_values[epoch_str][name]['step'] is None:
                self.metric_values[epoch_str][name]['step'] = []
            
            self.metric_values[epoch_str][name]['step'].append(value)
        
        if on_epoch:
            if self.metric_values[epoch_str][name]['epoch'] is None:
                self.metric_values[epoch_str][name]['epoch'] = 0
                self.metric_values[epoch_str][name]['num_elements'] = 0
            
            self.metric_values[epoch_str][name]['epoch'] += (value * batch_size)
            self.metric_values[epoch_str][name]['num_elements'] += batch_size

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
    
    def reset_grad(self):
        pass

    def backprop(self):
        pass
    
    def configure_optimizers(self):
        pass

    def init_scheduler(self, scheduler, scheduler_params):
        pass
