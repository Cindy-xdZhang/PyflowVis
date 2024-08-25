import numpy as np
import torch
from .transforms_factory import DataTransforms


@DataTransforms.register_module()
class ToTensor(object):
    def __init__(self, **kwargs):
        pass
    def __call__(self, data):
        if not torch.is_tensor(data):
            if str(data.dtype) == 'float64':
                data= data.astype(np.float32)
            data = torch.from_numpy(np.array(data))
        return data


@DataTransforms.register_module()
class MinMaxNormalization(object):
    def __init__(self, minV,maxV, **kwargs):
        self.minV = minV
        self.maxV=  maxV
        self.one_divided_by_range=1.0/(maxV-minV)

    def __call__(self, data):
        data=(data-min)*self.one_divided_by_range
        return data

@DataTransforms.register_module()    
class WhiteNoise(object):
    def __init__(self, minV,maxV,noiseMaginitude,**kwargs):
        self.range0=minV*noiseMaginitude
        self.range1=maxV*noiseMaginitude
       
    def __call__(self, tensor):
        noise = torch.empty_like(tensor).uniform_(self.range0, self.range1).float()
        # Add the white noise to the sample
        sample_with_noise = tensor + noise
        return sample_with_noise

    
@DataTransforms.register_module()
class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0,**kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.normal(mean=self.mean, std=self.std, size=tensor.size()).float()
        # Add the Gaussian noise to the sample
        sample_with_noise = tensor + noise
        return sample_with_noise




# @DataTransforms.register_module()
# class RandomJitter(object):
#     def __init__(self, jitter_sigma=0.01, jitter_clip=0.05, **kwargs):
#         self.noise_sigma = jitter_sigma
#         self.noise_clip = jitter_clip

#     def __call__(self, data):
#         jitter = np.clip(self.noise_sigma * np.random.randn(data['pos'].shape[0], 3), -self.noise_clip, self.noise_clip)
#         data['pos'] += jitter
#         return data


# @DataTransforms.register_module()
# class RandomShift(object):
#     def __init__(self, shift=[0.2, 0.2, 0], **kwargs):
#         self.shift = shift

#     def __call__(self, data):
#         shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
#         data['pos'] += shift
#         return data

#     def __repr__(self):
#         return 'RandomShift(shift_range: {})'.format(self.shift_range)

