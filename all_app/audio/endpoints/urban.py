import io
from fastapi import APIRouter
import torch
import torch.nn as nn
from torchaudio import transforms
import uvicorn
import torch.nn.functional as F
import soundfile as sf
import streamlit as st

urban_audio = APIRouter(prefix='/urban', tags=['Urban'])


class CheckAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(CheckAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.flatten = nn.Flatten()

        self.second = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)

max_len = 500

urban = torch.load('labels_urban.pth')
index_to_labels = {ind: lab for ind, lab in enumerate(urban)}

model = CheckAudio()
model.load_state_dict(torch.load('model_urban.pth', map_location=device))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform)

    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sr != 22050:
        resample = transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform)

    if spec.shape[-1] > max_len:
        spec = spec[:, :, :max_len]
    if spec.shape[-1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[-1]))

    return spec

def urban_predict():

    st.title('👾Urban')
    st.text('Загрузите аудиофайл (.wav) для распознавания команды')
    file = st.file_uploader('Выберите аудиофайл', type=['wav'])

    if not file:
        st.info('Загрузите аудиофайл')
    else:
        st.audio(file)

        if st.button('🔍Распознать'):
            try:
                data = file.read()
                wf, sr = sf.read(io.BytesIO(data), dtype='float32')
                wf = torch.from_numpy(wf).T if not isinstance(wf, torch.Tensor) else wf

                spec = change_audio(wf, sr)
                spec = spec.unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(spec)
                    pred_ind = torch.argmax(y_pred, dim=1).item()
                    pred_class = index_to_labels[pred_ind]

                st.write({f'индекс: {pred_ind}, жанр: {pred_class}'})

            except Exception as e:
                st.exception(f'{e}')

