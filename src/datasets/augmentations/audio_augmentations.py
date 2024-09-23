import src.utils as utils
import torch

class AudioAugmentations():
    def __init__(self, augmentations_list) -> None:
        
        self.augmentations = []
        self.augmentation_prob = []
        
        for augmentation_desc in augmentations_list:
            assert 'type' in augmentation_desc, 'Augmentation has no specified type!'
            assert 'prob' in augmentation_desc, 'Augmentation has no specified probability!'

            # If no params are specified, assume there are no params given
            if 'params' not in augmentation_desc:
                augmentation_desc['params'] = {}

            augmentation = utils.import_attr(augmentation_desc['type'])(**augmentation_desc['params'])
            self.augmentations.append(augmentation)
            self.augmentation_prob.append(augmentation_desc['prob'])

    def apply_random_augmentations(self, input_audio, gt_audio):
        augmented_input_audio = input_audio
        augmented_gt_audio = gt_audio

        # Go over perturbations
        for prob, augmentation in zip(self.augmentation_prob, self.augmentations):
            # With some probability, apply this perturbation
            if torch.rand((1,)).item() < prob:
                augmented_input_audio, augmented_gt_audio = \
                    augmentation(augmented_input_audio, augmented_gt_audio)
        
        return augmented_input_audio, augmented_gt_audio
