from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess external image
img_path = 'test2.jpg'  # replace with your image path
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # normalize
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Predict class
pred = model.predict(img_array)
predicted_class = np.argmax(pred)

# Display result
plt.imshow(img)
plt.title(f"Predicted Class: {class_names[predicted_class]}")
plt.axis('off')
plt.show()
