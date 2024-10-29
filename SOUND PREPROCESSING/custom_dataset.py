from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import torch


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, max_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.max_samples = max_samples
        
    
    def __len__(self):
        return len(self.annotations)
    
    def is_multichannel(self,signal):
        if signal.shape[0] > 1:
            return True
        else:
            return False


    def multi_to_single(self,signal):
        if self.is_multichannel(signal):
            signal = torch.mean(signal,dim = 0)
            return signal
        else:
            return signal
        
    
    def get_audio_sample(self,index):
        folder = f'fold{self.annotations.iloc[index, 5]}'
        path = os.path.join(self.audio_dir,folder,self.annotations.iloc[index, 0])
        return path
    
    def get_audio_label(self, index):
        label = self.annotations.iloc[index, 6]
        return label

    def resample_if_necessary(self, sr, signal):
        if sr == self.target_sample_rate:
            return signal
        else:
            resampling = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            return resampling(signal)
        
    def get_pad_or_cut(self, signal):
        if signal.shape[0] > self.max_samples:
            signal = signal[:self.max_samples]
            return signal
        
        elif signal.shape[0] < self.max_samples:
            num_missing_samples = self.max_samples - signal.shape[0]
            signal = torch.nn.functional.pad(signal,(0, num_missing_samples))
            return signal
        else:
            return signal
            

    def __getitem__(self,index):
        audio_sample_path = self.get_audio_sample(index)
        label = self.get_audio_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self.resample_if_necessary(sr,signal)
        signal = self.multi_to_single(signal)
        signal = self.get_pad_or_cut(signal)
        signal = signal.to(device=self.device)        
        signal = self.transformation(signal)
        return signal, label
    
