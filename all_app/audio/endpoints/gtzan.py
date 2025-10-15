import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from fastapi import APIRouter
import streamlit as st
import soundfile as sf

audio_gtzan = APIRouter(prefix='/gtzan', tags=['GTZAN'])

class CheckMelodia(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64,
)
max_len = 500

genres = torch.load('labels_gtzan.pth')
index_to_label = {ind: lab for ind, lab in enumerate(genres)}

model = CheckMelodia()
model.load_state_dict(torch.load('model_gtzan.pth', map_location=device))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if waveform.ndim > 1:
        waveform = torch.mean(torch.tensor(waveform), dim=0, keepdim=True)
    else:
        waveform = torch.tensor(waveform).unsqueeze(0)

    if sr != 22050:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform)

    if spec.shape[-1] > max_len:
        spec = spec[:, :, :max_len]
    if spec.shape[-1] < max_len:
        count_len = max_len - spec.shape[-1]
        spec = F.pad(spec, (0, count_len))

    return spec


def gtzan_predict():
    st.title('🎵 GTZAN Genre Classifier')
    st.text('Загрузите аудиофайл (.wav), чтобы распознать жанр 🎧')

    file = st.file_uploader('Выберите аудиофайл', type=['wav'])

    if file is None:
        st.info('Пожалуйста, загрузите .wav файл')
    else:
        st.audio(file)

        if st.button('🔍 Распознать жанр'):
            try:
                data = file.read()
                wf, sr = sf.read(io.BytesIO(data), dtype='float32')
                spec = change_audio(wf, sr).to(device)

                # add batch dimension [B, C, H, W]
                spec = spec.unsqueeze(0)

                with torch.no_grad():
                    y_pred = model(spec)
                    pred_ind = torch.argmax(y_pred, dim=1).item()
                    pred_class = index_to_label[pred_ind]

                st.success({f"индекс": pred_ind, "жанр": pred_class})

            except Exception as e:
                st.exception(f'{e}')
