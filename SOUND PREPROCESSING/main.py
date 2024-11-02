from custom_dataset import UrbanSoundDataset
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":
    annotations_file = "C:/ANKITH/SOUND/SOUND PREPROCESSING/UrbanSound8K/metadata/UrbanSound8K.csv"
    audio_dir = "C:/ANKITH/SOUND/SOUND PREPROCESSING/UrbanSound8K/audio"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print(f'Device using :{device}')
    new_sr = 8000
    max_samples = 16000

    
    
    #melspect = torchaudio.transforms.MelSpectrogram(sample_rate=new_sr,n_fft=1024,hop_length=512,n_mels=64)
    # usd = UrbanSoundDataset(annotations_file = annotations_file, 
    #                         audio_dir = audio_dir,
    #                         transformation = melspect,
    #                         target_sample_rate = new_sr, 
    #                         max_samples = max_samples, 
    #                         device = device)
    

    print(f'Total number of samples in dataset :{len(usd)}')
    signal, label = usd[1]
    print(signal.shape)