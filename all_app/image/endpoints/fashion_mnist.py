from fastapi import APIRouter
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

fashion_router = APIRouter(prefix='/fashion', tags=['FashionMnist'])


class FashionCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )

  def forward(self, x):
    x = self.conv_block(x)
    x = self.fc(x)
    return x


model = FashionCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model_fashion.pth', map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Ankle boot']

def fashion_predict():

    st.title('👘Fashion MNIST')
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
                st.write(f'answer_class: {classes[predicted]}')

            except Exception as e:
                st.exception(f'Error: str{e}')


