import copy
import math
import os
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Type, Union

import numpy as np
import matplotlib.pyplot as plt

import torch
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics import MeanMetric

from configurationupdate import Config
from loss import PerceptualLoss


class Consistency(LightningModule):
    """ Consistency model implementation based on: https://github.com/openai/consistency_models """

    def __init__(
        self,
        config: Config,
        bins_min: int = 2,
        bins_max: int = 150,
        bins_rho: float = 7,
        loss_func: str = 'LPIPS',
        initial_ema_decay: float = 0.9,
        optimizer_type: Type[optim.Optimizer] = optim.RAdam,
        num_samples: int = 16,
        use_ema: bool = True,
        sample_seed: int = 0,
        **kwargs,
    ) -> None:

        """
        Args:
            config: Network configuration.
            bins_min: Minimum number of time steps.
            bins_max: Maximum number of time steps.
            bins_rho: Determines time boundaries.
            loss_func: Loss function.
            initial_ema_decay: Exponential average decay parameter.
            optimizer_type: Gradient decent optimizer.
            num_samples: Number of generated samples per batch.
            use_ema: Enables the EMA model for inference.
            sample_seed: Seed value of the random number generator.
        """

        super().__init__()
        
        self.save_hyperparameters(ignore=['loss_fn'])

        self.config = config

        # configuration `in_channels` should equal the total number of
        # channels fed to the UNet.  earlier code mistakenly added the
        # output channels again, which doubled the channel count during
        # model construction and caused mismatches during forward passes.
        model = UNet2DModel(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            block_out_channels=self.config.channels,
            down_block_types=self.config.down_block_types,
            up_block_types=self.config.up_block_types
        )

        self.model = model
        self.model_ema = copy.deepcopy(model)
        self.image_size = self.config.sample_dimension

        self.model_ema.requires_grad_(False)

        if loss_func == "LPIPS":
            self.loss_fn = PerceptualLoss(net_type="squeeze")
        elif loss_func == "MSE":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"loss function not defined: {loss_func}")

        self.optimizer_type = optimizer_type

        self.learning_rate = self.config.lr
        self.initial_ema_decay = initial_ema_decay

        self.data_std = self.config.data_std
        self.time_min = self.config.time_min 
        self.time_max = self.config.time_max
        self.clip = self.config.clip_output

        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self._train_loss_tracker = MeanMetric()
        self._val_loss_tracker = MeanMetric()
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()

        self.num_samples = num_samples
        self.use_ema = use_ema
        self.sample_seed = sample_seed
        self.sample_steps = 1

     
    def configure_optimizers(self):
        optimizer = self.optimizer_type(self.parameters(), lr=self.learning_rate)

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor):

        return self._forward(self.model, images, times)

    def _forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        times: torch.Tensor):
        """ Evaluates the network

        Args:
            model: network modul
            images: Input batch 
            times: Noise time

        Returns:
            Network output 
        """

        skip_coef = self.data_std**2 / ((times - self.time_min).pow(2) + self.data_std**2)
        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        out: UNet2DOutput = model(images, times)
        noisy_target_part = images[:, -self.config.out_channels:]

        out = self.image_time_product(noisy_target_part, skip_coef) + self.image_time_product(out.sample, out_coef)

        if self.clip:
            return out.clamp(-1.0, 1.0)

        return out


    def training_step(self,
                      batch,
                      *args,
                      **kwargs):
        """ Performs a single training step.

        Args:
            images: Input batch 

        Returns:
            Training loss
        """

        era5, images = batch
        _bins = self.bins

        noise = torch.randn(images.shape, device=images.device)
        timesteps = torch.randint(
            0,
            _bins - 1,
            (images.shape[0],),
            device=images.device,
        ).long()

        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)

        current_noise_image = images + self.image_time_product(
            noise,
            current_times,
        )

        next_noise_image = images + self.image_time_product(
            noise,
            next_times,
        )
        era5_up = torch.nn.functional.interpolate(
            era5, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            current_input = torch.cat([era5_up, current_noise_image], dim=1)
            target = self._forward(
                self.model_ema,
                current_input,
                current_times,
            )

        model_input = torch.cat([era5_up, next_noise_image], dim=1)

        pred = self(model_input, next_times)    

        loss = self.loss_fn(pred, target)

        self._train_loss_tracker(loss)
        self.log(
            "train_loss",
            self._train_loss_tracker,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True
        )

        self._bins_tracker(_bins)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True
        )

        return loss


    @torch.no_grad()
    def validation_step(self,
                        batch,
                        *args,
                        **kwargs)-> torch.Tensor:
        """ Performs a single validation step.

        Args:
            images: Input batch 

        Returns:
            Validation loss
        """

        era5, images = batch  
        _bins = self.bins

        noise = torch.randn(images.shape, device=images.device)
        timesteps = torch.randint(
            0,
            _bins - 1,
            (images.shape[0],),
            device=images.device,
        ).long()

        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)

        current_noise_image = images + self.image_time_product(
            noise,
            current_times,
        )

        next_noise_image = images + self.image_time_product(
            noise,
            next_times,
        )
        era5_up = torch.nn.functional.interpolate(
            era5, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            current_input = torch.cat([era5_up, current_noise_image], dim=1)
            target = self._forward(
                self.model_ema,
                current_input,
                current_times,
            )

        model_input = torch.cat([era5_up, next_noise_image], dim=1)

        pred = self(model_input, next_times)    

        loss = self.loss_fn(pred, target)

        self._val_loss_tracker(loss)
        self.log(
            "val_loss",
            self._val_loss_tracker,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True
        )


    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update()


    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        
        


    @property
    def ema_decay(self):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins)


    @property
    def bins(self) -> int:
        return math.ceil(
            math.sqrt(
                self.trainer.global_step
                / self.trainer.estimated_stepping_batches
                * (self.bins_max**2 - self.bins_min**2)
                + self.bins_min**2
            )
        )


    def timesteps_to_times(self,
                           timesteps: torch.LongTensor,
                           bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )


    @torch.no_grad()
    def sample(
        self,
        num_samples: Optional[int] = 16,
        steps: Optional[int] = 1,
        x_image_size: Optional[int] = None,
        y_image_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: Optional[bool] = False,
    ) -> torch.Tensor:
        """ Unconditioned sampler. 

        Args:
            num_samples: Number of generated samples per batch.
            steps: Number of sample steps.
            x_image_size: Outuput dimension in height direction.
            y_image_size: Outuput dimension in width direction.
            generator: Noise generator.
            use_ema: Enables EMA model.

        Returns:
            Generated batch of samples.
        """

        # unconditioned sampling uses full `in_channels` as defined in config
        if x_image_size and y_image_size is not None:
            shape = (num_samples, self.config.in_channels, x_image_size, y_image_size)
        else:
            shape = (num_samples, self.config.in_channels, self.config.sample_dimension[0], self.config.sample_dimension[1])

        time = torch.tensor([self.time_max], device=self.device)

        images: torch.Tensor = self._forward(
            self.model_ema if use_ema else self.model,
            randn_tensor(shape, generator=generator, device=self.device) * time,
            time,
        )

        if steps <= 1:
            return images

        _timesteps = list(
            reversed(range(0, self.bins_max, self.bins_max // steps - 1))
        )[1:]
        _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

        times = self.timesteps_to_times(
            torch.tensor(_timesteps, device=self.device), bins=150
        )

        for time in times:
            noise = randn_tensor(shape, generator=generator, device=self.device)
            images = images + math.sqrt(time.item() ** 2 - self.time_min**2) * noise
            images = self._forward(
                self.model_ema if use_ema else self.model,
                images,
                time[None],
            )

        return images


    @torch.no_grad()
    def sample_conditional(
        self,
        conditioning,
        x_image_size,
        y_image_size,
        steps: int = 1,
        sample_times: Optional[List[float]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:

        """ Conditioned sampler. 

        Args:
            conditioning: Image serving as the conditioning.
            x_image_size: Outuput dimension in height direction.
            y_image_size: Outuput dimension in width direction.
            steps: Number of sample steps.
            sample_times: Noising time.
            generator: Noise generator.
            use_ema: Enables EMA model.

        Returns:
            Generated batch of samples.
        """

        # conditioning expected to be 4‑D (B,cond_channels,H,W)
        if conditioning.dim() == 5 and conditioning.size(1) == 1:
            conditioning = conditioning.squeeze(1)
        if conditioning.dim() != 4:
            raise ValueError(f"conditioning must be 4‑D (B,C,H,W), got {conditioning.shape}")

        if sample_times is not None and len(sample_times) > 0:
            time = torch.tensor([sample_times[0]], device=self.device)
        else:
            time = torch.tensor([self.time_max], device=self.device)

        # build initial high‑res input by concatenating conditioning with
        # zeros for the target channels.  here `config.in_channels` is the
        # full width (cond + output); compute cond channels explicitly.
        cond_ch = self.config.in_channels - self.config.out_channels
        if conditioning.shape[1] != cond_ch:
            raise ValueError(
                f"conditioning has {conditioning.shape[1]} channels, "
                f"expected {cond_ch}"
            )
        B, Cc, Hc, Wc = conditioning.shape
        
        # Interpolate conditioning to target size
        conditioning_up = torch.nn.functional.interpolate(
            conditioning, size=(x_image_size, y_image_size), mode="bilinear", align_corners=False
        )

        # KRITISCH: Noise nur auf die Target-Kanäle addieren, NOT auf das Conditioning!
        # Während des Trainings ist era5_up immer sauber — das Modell erwartet das auch bei Inference.
        target_noise_shape = (B, self.config.out_channels, x_image_size, y_image_size)
        target_noise = randn_tensor(target_noise_shape, generator=generator, device=self.device)
        noisy_target = target_noise * time  # zeros + noise*t

        images = torch.cat([conditioning_up, noisy_target], dim=1)

        images_cond = images.clone()

        # forward expects all channels; output is (B, out_channels, H, W)
        images: torch.Tensor = self._forward(self.model_ema if use_ema else self.model, images, time)

        if sample_times is not None and len(sample_times) > 1:
            times = [torch.tensor([t], device=self.device) for t in sample_times[1:]]
        else:
            if steps <= 1:
                return images, images_cond
            _timesteps = list(
                reversed(range(0, self.bins_max, self.bins_max // steps - 1))
            )[1:]
            _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

            times = self.timesteps_to_times(torch.tensor(_timesteps, device=self.device), bins=150)

        for time in times:
            # Nur die Target-Kanäle werden re-verrauscht (Conditioning bleibt sauber)
            target_noise = randn_tensor(
                (B, self.config.out_channels, x_image_size, y_image_size),
                generator=generator,
                device=self.device,
            )
            images = images + math.sqrt(time.item() ** 2 - self.time_min**2) * target_noise

            if sample_times is None:
                time = time[None]

            # Conditioning wieder concatenieren vor dem nächsten Forward-Pass
            images_full = torch.cat([conditioning_up, images], dim=1)
            images = self._forward(
                self.model_ema if use_ema else self.model,
                images_full,
                time,
            )

        return images, images_cond


    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)
#