from PIL import Image
from keras.models import load_model
import numpy as np

# Predict the class of an image
def predict(image_path):
    #load the model
    model = load_model('models/model50epoach.h5')
    # model = load_model('models/model.h5')
    img = Image.open(image_path).convert('L').resize((28, 28), Image.ANTIALIAS)
    img = np.array(img)
    results =model.predict(img[None,:,:])
    print(np.argmax(results))
    dict = {"0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J", "10": "K","11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T", "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"}
    print(dict[str(np.argmax(results))])
    return dict[str(np.argmax(results))]

if __name__ == '__main__':
    predict("input_samples/nn.jpg")


