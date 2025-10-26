import streamlit as st
import torch
import torch.nn as nn
from fastapi import APIRouter
from torchvision import transforms
from PIL import Image

image_router = APIRouter(prefix='/image', tags=['Group-Image'])


class CheckImageAlexNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 19)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = CheckImageAlexNET()
model.load_state_dict(torch.load("my_model.pth", map_location=device))
model.to(device)
model.eval()


class_name = [
    'Pebbles',
    'Shells',
    'airplane',
    'bear',
    'bike',
    'car',
    'cat',
    'dog',
    'elephant',
    'helicopter',
    'horse',
    'laptop',
    'lion',
    'lower_clothes',
    'panda',
    'phone',
    'scooter',
    'ship',
    'upper_clothes'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def group_predict():
    st.title("🧠 ALL Dataset Image Classifier")
    st.text("Загрузите изображение, модель определит его класс")

    file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        if st.button("Определить класс"):
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(img_tensor)
                pred_ind = torch.argmax(y_pred, dim=1).item()

                pred_class = class_name[pred_ind]
                st.success(f" Предсказанный класс: {pred_class}")
