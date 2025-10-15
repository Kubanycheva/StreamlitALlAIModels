from fastapi import APIRouter
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

cifar_router = APIRouter(prefix='/cifar100', tags=['Cifar100'])

class CheckImageCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckImageCnn()
model.load_state_dict(torch.load('cifar100_model.pth', map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
    'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def cifar_predict():

    st.title('😮‍💨CIFAR100')
    st.text('Загрузите изображение с цифрой, и модель попробует ее распознать')

    file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])
    if file is None:
        st.info('Выбер правильный файл')
    else:
        st.image(file, caption='Загруженное изображение')

        if st.button('🔍Определить изображение'):
            try:
                image_data = file.read()
                img = Image.open(io.BytesIO(image_data)).convert("RGB")  # RGB
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(img_tensor)
                    predicted_index = torch.argmax(y_pred, dim=1).item()
                    predicted_class = classes[predicted_index]

                st.write({"predicted_class": {predicted_class}})

            except Exception as e:
                st.exception(f'Error: str{e}')

