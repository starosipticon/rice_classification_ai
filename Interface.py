import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load your trained model
model = keras.models.load_model('C:/PythonProject/AI_Project/Model.h5')

# Define the class labels
class_labels = {
    0: 'бактериальная пятнистость листьев', 
    1: 'бактериальная полосатость листьев', 
    2: 'бактериальная пятнистость соцветия', 
    3: 'бласт', 
    4: 'коричневые пятна', 
    5: 'мертвое сердце', 
    6: 'плесневелая пятнистость', 
    7: 'хиспа', 
    8: 'нет болезней', 
    9: 'тунгро'
    # Add more labels as needed
}

# Функция предобработки изображения перед передачей его модели
def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Функция предсказания заболевания
def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    disease_label = class_labels[predicted_class]
    return disease_label

# Функция открытия диалога выбора изображения
def select_image():
    file_path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Изображения", "*.png;*.jpg;*.jpeg")])
    if file_path:
        display_image(file_path)
        disease_label = predict_disease(file_path)
        result_label.config(text=f"Заболевание: {disease_label}")

# Функция отображения выбранного изображения
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Создание основного окна приложения
app = tk.Tk()
app.title("Обнаружение болезней риса")

# Установка размеров окна
app.geometry("400x500")

# Создание и настройка виджетов
select_button = tk.Button(app, text="Выбрать изображение", command=select_image)
image_label = tk.Label(app)
result_label = tk.Label(app, text="Заболевание: ")

# Размещение виджетов в окне
select_button.pack(pady=10)
image_label.pack(pady=10)
result_label.pack(pady=10)

# Запуск приложения
app.mainloop()
