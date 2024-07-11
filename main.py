import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageTk

# Define the animal classes you are interested in
animal_classes = {
    'dog', 'cat', 'lion', 'tiger', 'bear', 'elephant', 'monkey', 'horse', 'sheep', 'cow', 'goat', 'deer', 'bird', 'fish',
    'reptile', 'insect', 'snake', 'spider', 'frog', 'turtle', 'hamster', 'guinea_pig', 'rabbit', 'chicken', 'rooster',
    'penguin', 'dolphin', 'whale', 'shark', 'octopus', 'crab', 'lobster', 'bee', 'butterfly', 'ant', 'bat', 'buffalo',
    'camel', 'cheetah', 'chimpanzee', 'crocodile', 'donkey', 'eagle', 'flamingo', 'giraffe', 'goose', 'hawk', 'hyena',
    'jaguar', 'kangaroo', 'koala', 'leopard', 'lizard', 'lynx', 'mole', 'moose', 'ostrich', 'otter', 'owl', 'panda',
    'parrot', 'peacock', 'pelican', 'pigeon', 'platypus', 'polar_bear', 'porcupine', 'raccoon', 'rat', 'raven', 'rhinoceros',
    'scorpion', 'seal', 'seahorse', 'skunk', 'sloth', 'snail', 'squid', 'squirrel', 'swan', 'vulture', 'walrus', 'wombat',
    'woodpecker', 'yak', 'zebra'
}

# Load the pre-trained model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")

# Load labels from TensorFlow datasets
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    # Run the image through the model
    predictions = model(image)
    predicted_class = np.argmax(predictions[0], axis=-1)
    predicted_label = labels[predicted_class]
    confidence_score = np.max(tf.nn.softmax(predictions[0])) * 100
    
    return predicted_label, confidence_score

def identify_animal(image_path):
    predicted_label, confidence_score = classify_image(image_path)
    
    if predicted_label.lower() in animal_classes:
        return f'The image is classified as a {predicted_label} with {confidence_score:.2f}% confidence.'
    else:
        return f'The image is classified as a {predicted_label} with {confidence_score:.2f}% confidence, which is not in the predefined animal classes.'

def upload_action(event=None):
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)
            
            img_label.config(image=photo)
            img_label.image = photo
            result = identify_animal(file_path)
            result_label.config(text=result)
            log_classification(file_path, result)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def log_classification(file_path, result):
    log_list.insert(tk.END, f"{file_path.split('/')[-1]}: {result}")

# Create the main window
root = tk.Tk()
root.title("Animal Classifier")
root.geometry("500x800")
root.configure(bg='#f0f0f0')

# Create and place the widgets
title_label = Label(root, text="Animal Classifier", font=('Helvetica', 18, 'bold'), bg='#f0f0f0')
title_label.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=upload_action, font=('Helvetica', 14), bg='#4CAF50', fg='white')
upload_button.pack(pady=10)

img_label = Label(root, bg='#f0f0f0')
img_label.pack(pady=10)

result_label = Label(root, text="", font=('Helvetica', 14), bg='#f0f0f0', wraplength=400, justify="center")
result_label.pack(pady=10)

log_label = Label(root, text="Classification Log", font=('Helvetica', 14, 'bold'), bg='#f0f0f0')
log_label.pack(pady=10)

log_list = tk.Listbox(root, width=50, height=10)
log_list.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
