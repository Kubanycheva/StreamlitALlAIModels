from fastapi import APIRouter
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

check_mnist = APIRouter(prefix='/mnist', tags=['Mnist'])


class CheckNumber(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckNumber().to(device)
model.load_state_dict(torch.load("image_mnist.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

classes = [
    "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
    "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"
]

def mnist_predict():

    st.title('📎MNIST Classifier')
    st.text('Загрузите изображение с цифрой, и модель попробует ее распознать')

    file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg', 'svg'])

    if file is None:
        st.info('Выбери правильный файл')
    else:
        st.image(file, caption='Загруженное изображение')

        if st.button('🔍Опредилить цифру'):
            try:
                image_data = file.read()
                img = Image.open(io.BytesIO(image_data))
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(img_tensor)
                    predicted = torch.argmax(y_pred, dim=1).item()
                st.success(f'answer_class: {predicted}')

            except Exception as e:
                st.exception(f'Error: {e}')


