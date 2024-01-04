from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
pneumonia_model_path = 'models/pneu_cnn_model.h5'
malaria_model_path = 'models/malaria_detect.h5'
pneumonia_model = load_model(pneumonia_model_path)
malaria_model = load_model(malaria_model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    imagePath = None
    tips = None

    if request.method == 'POST':
        imagefile = request.files["imagefile"]
        image_path = './static/' + imagefile.filename
        imagefile.save(image_path)

        model_selection = request.form.get("modelSelection")

        if model_selection == 'pneumonia':
            img = image.load_img(image_path, target_size=(500, 500), color_mode='grayscale')
            x = image.img_to_array(img)
            x = x / 255
            x = np.expand_dims(x, axis=0)
            classes = pneumonia_model.predict(x)
            prediction = classes[0][0] * 100  # Convert prediction to percentage
            tips = f"This result indicates a potential presence of pneumonia with confidence {prediction:.2f}%." if prediction >= 50 else "Great news! The result is negative. To maintain good respiratory health, consider the following tips: [Insert prevention tips here for pneumonia]."
        elif model_selection == 'malaria':
            img = image.load_img(image_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            classes = malaria_model.predict(x)
            prediction = classes[0][0]
            tips = "This result indicates a potential presence of malaria." if prediction >= 0.5 else "Great news! The result is negative. To prevent malaria, consider the following tips: [Insert prevention tips here for malaria]."
        else:
            # Handle invalid model selection
            return render_template('index1.html')

    return render_template('index1.html', prediction=prediction, imagePath=image_path, tips=tips)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
