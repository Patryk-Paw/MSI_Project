import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model("fruit_classifier_model.h5")


class_names = [
    "abiu", "acai", "acerola", "ackee", "ambarella", "apple", "apricot", "avocado",
    "banana", "barbadine", "barberry", "bitter_gourd", "black_berry", "black_mullberry",
    "brazil_nut", "betel_nut", "camu_camu", "cashew", "cempedak", "cherimoya",
    "chenet", "chico", "chokeberry", "cluster_fig", "coconut", "corn_kernel", "cranberry",
    "cupuacu", "custard_apple", "damson", "dewberry", "dragonfruit", "durian", "eggplant",
    "elderberry", "emblic", "feijoa", "fig", "finger_lime", "gooseberry", "grape",
    "grapefruit", "greengage", "grenadilla", "goumi", "guava", "hard_kiwi", "hawthorn",
    "hog_plum", "horned_melon", "indian_strawberry", "jackfruit", "jaboticaba", "jalapeno",
    "jamaica_cherry", "jambul", "jujube", "jocote", "kaffir_lime", "kumquat", "lablab",
    "langsat", "longan", "malay_apple", "mabolo", "mango", "mandarine", "medlar",
    "mock_strawberry", "morinda", "mountain_soursop", "olive", "oil_palm", "otaheite_apple",
    "papaya", "passion_fruit", "pawpaw", "pea", "pineapple", "plumcot", "pomegranate",
    "prikly_pear", "quince", "rambutan", "raspberry", "redcurrant", "rose_hip",
    "rose_leaf_bramble", "salak", "santol", "sapodilla", "sea_buckthorn", "strawberry_guava",
    "sugar_apple", "taxus_baccata", "ugli_fruit", "white_currant", "yali_pear"
]

IMG_SIZE = (64, 64)

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy() * 100

    return predicted_class, confidence

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, confidence = predict_image(file_path)
        result_label.config(text=f"Owoc: {predicted_class}\nPewność: {confidence:.2f}%")

        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img


root = tk.Tk()
root.title("🍇 Rozpoznawanie owoców")

open_button = tk.Button(root, text="Wybierz zdjęcie owocu", command=open_file)
open_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
