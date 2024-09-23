import torch


class PeakNormAugmentation:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, audio_data, gt_audio):
        peak = torch.abs(audio_data).max()
        
        scale = torch.randn((1,)).item() * (self.max_scale - self.min_scale) + self.min_scale
        scale = scale / (peak + 1e-6)

        augmented_audio_data = audio_data * scale
        augmented_gt_audio = gt_audio * scale
        
        return augmented_audio_data, augmented_gt_audio