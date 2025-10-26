import streamlit as st
from all_app.image.endpoints import mnist, fashion_mnist, cifar100, group_image
from all_app.audio.endpoints import gtzan, urban, speech
from all_app.text.endpoints import news

st.title('Welcome')

with st.sidebar:
    st.header('Menu')
    name = st.radio('Выбери', ['Gtzan', 'Speech', 'Urban', 'Cifar100', 'Mnist', 'Fashion-Mnist', 'Ag-News', 'Group-Image'])

if name == 'Mnist':
    mnist.mnist_predict()
elif name == 'Fashion-Mnist':
    fashion_mnist.fashion_predict()
elif name == 'Cifar100':
    cifar100.cifar_predict()
elif name == 'Gtzan':
    gtzan.gtzan_predict()
elif name == 'Urban':
    urban.urban_predict()
elif name == 'Speech':
    speech.speech_predict()
elif name == 'Ag-News':
    news.news_predict()
elif name == 'Group-Image':
    group_image.group_predict()