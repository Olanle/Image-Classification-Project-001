Project-001-Image-Classification

Here is a simple documentation for your code:

---

### **Project Documentation: Image Classification Using CNN (CIFAR-10 Dataset)**

#### **Overview**

This code demonstrates a basic image classification pipeline using TensorFlow and Keras. The task is to classify images from the CIFAR-10 dataset into 10 different categories (e.g., airplane, car, bird, etc.). The pipeline includes data preprocessing, model creation, training, and evaluation, followed by image predictions and visualization.

#### **Libraries Used**

* **TensorFlow**: The main library for creating and training the Convolutional Neural Network (CNN).
* **NumPy**: Used for numerical operations and data manipulation.
* **Matplotlib**: For plotting and visualizing images and predictions.
* **PIL (Python Imaging Library)**: For image loading and preprocessing.
* **Google Colab**: For cloud-based environment and file uploading.

#### **Code Explanation**

1. **Installing Dependencies**
   The following libraries are installed:

   * TensorFlow
   * NumPy
   * Matplotlib

   ```python
   %pip install tensorflow numpy matplotlib
   ```

2. **Loading and Normalizing the Dataset**
   The CIFAR-10 dataset is loaded using `tf.keras.datasets.cifar10.load_data()`. The pixel values are normalized between -1 and 1 to help with model training.

   ```python
   (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
   train_images = train_images / 127.5 - 1
   test_images = test_images / 127.5 - 1
   ```

3. **Model Creation (CNN)**
   A simple CNN model is built using `tf.keras.models.Sequential()`:

   * **Conv2D layers**: For feature extraction.
   * **MaxPooling2D layers**: To reduce spatial dimensions.
   * **Flatten layer**: To flatten the output for dense layers.
   * **Dense layers**: For classification.

   ```python
   model = models.Sequential([
       layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
       layers.MaxPooling2D((2, 2), strides=(2, 2)),
       layers.Conv2D(128, (3, 3), activation='relu'),
       layers.MaxPooling2D((2, 2), strides=(2, 2)),
       layers.Flatten(),
       layers.Dense(128, activation='relu'),
       layers.Dense(84, activation='relu'),
       layers.Dense(10, activation='softmax')
   ])
   ```

4. **Model Compilation and Training**
   The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric. It is then trained for 10 epochs.

   ```python
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
   ```

5. **Model Evaluation**
   After training, the model is evaluated on the test dataset to determine its accuracy.

   ```python
   test_loss, test_accuracy = model.evaluate(test_images, test_labels)
   print(f'Accuracy on {test_images.shape[0]} test images: {test_accuracy * 100:.2f}%')
   ```

6. **Image Classification and Visualization**
   A function `view_classification` is defined to display the image and its predicted class probabilities. The model makes a prediction on a specific image from the test dataset.

   ```python
   def view_classification(image, probabilities):
       fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
       denormalized_image = (image + 1) / 2
       ax1.imshow(denormalized_image)
       ax1.axis('off')
       ax2.barh(np.arange(10), probabilities)
       ax2.set_aspect(0.1)
       ax2.set_yticks(np.arange(10))
       ax2.set_yticklabels(classes)
       ax2.set_title('Class Probability')
       ax2.set_xlim(0, 1.1)
       plt.tight_layout()
   ```

7. **File Upload and Prediction**
   Users can upload an image for the model to classify. The uploaded image is preprocessed (converted to RGB, resized, and normalized), and the model predicts the class of the image. The predicted class and probabilities are displayed in a bar chart.

   ```python
   from google.colab import files
   from PIL import Image

   # Step 1: Upload image
   uploaded = files.upload()
   filename = list(uploaded.keys())[0]

   # Step 2: Load and preprocess image
   img = Image.open(filename).convert('RGB').resize((32, 32))  # Force RGB and resize
   img_array = np.array(img) / 255.0
   input_image = img_array.reshape(1, 32, 32, 3)

   # Step 3: Predict
   predictions = model.predict(input_image)
   predicted_class = np.argmax(predictions[0])

   # Step 4: Display image with predicted class
   plt.figure(figsize=(10, 4))

   # Show image
   plt.subplot(1, 2, 1)
   plt.imshow(img)
   plt.axis('off')
   plt.title(f"Predicted: {class_names[predicted_class]}")

   # Show probabilities as bar chart
   plt.subplot(1, 2, 2)
   plt.barh(class_names, predictions[0])
   plt.xlabel("Probability")
   plt.title("Class Probabilities")
   plt.tight_layout()
   plt.show()
   ```

8. **Displaying Probabilities**
   The code also displays the predicted classes with significant probabilities (above a threshold).

   ```python
   threshold = 0.01  # Adjust threshold as needed
   print("\nPredicted classes with significant probabilities:")
   for i, prob in enumerate(predictions[0]):
       if prob >= threshold:
           print(f"{class_names[i]}: {prob:.4f}")
   ```

#### **Results**

The model's accuracy after 10 epochs is approximately **71.09%** on the test dataset.

---

This is a basic structure for the documentation. Feel free to modify it to fit any additional details you might want to include.
