# Fashion MNIST Classification with Convolutional Neural Networks (CNN)

This project demonstrates the use of Convolutional Neural Networks (CNN) to classify images from the Fashion MNIST dataset. The goal is to accurately recognize different fashion items, such as shirts, trousers, and shoes, through a deep learning approach.

## Project Overview

This project involves the following key steps:

1. **Data Loading and Preprocessing**:
   - The Fashion MNIST dataset is loaded using Keras, consisting of 60,000 training images and 10,000 test images.
   - Images are resized and reshaped to fit the input shape required by the CNN model.

2. **Building the CNN Model**:
   - The CNN model is built using Keras with layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.
   - The architecture consists of multiple convolutional layers followed by pooling layers to extract and downsample features.

3. **Model Training and Evaluation**:
   - The model is compiled with categorical crossentropy loss and trained using the Adam optimizer.
   - The training process includes monitoring validation accuracy to assess model performance.

4. **Results**:
   - The model's performance is evaluated on the test set, achieving high accuracy in classifying various fashion items.

## Requirements

- Python 3.x
- Required libraries:
  - `tensorflow`
  - `keras`
  - `numpy`

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Fashion-MNIST-CNN.git
   cd Fashion-MNIST-CNN
   ```

2. Install the required libraries:

  ```bash
  pip install -r requirements.txt
  ```
3. Run the notebook:

Open the Fashion-MNIST-CNN.ipynb file in Jupyter Notebook or Jupyter Lab and run the cells sequentially.

## Key Results
- The CNN model effectively classified the Fashion MNIST dataset with high accuracy.
- The model leveraged multiple convolutional and pooling layers to extract features and reduce dimensionality.
- Dropout layers were used to prevent overfitting, improving the generalization of the model.

## Skills Learned
- Working on this project helped me develop and strengthen several key skills:
- Deep Learning with CNNs: Gained hands-on experience building, training, and evaluating Convolutional Neural Networks using Keras and TensorFlow.
- Image Data Preprocessing: Learned how to preprocess image data effectively, including reshaping, normalizing, and preparing datasets for CNN input.
- Model Optimization: Improved skills in tuning model architecture, including the use of dropout layers and pooling techniques to optimize performance and reduce overfitting.
- Evaluation Techniques: Developed an understanding of how to evaluate deep learning models using metrics like accuracy, and learned to monitor training and validation loss for performance assessment.
- Problem Solving: Enhanced problem-solving skills through debugging model architecture and adjusting parameters to achieve better results.
- Python and Keras: Strengthened my ability to use Python libraries such as TensorFlow and Keras for complex machine learning tasks.

## Conclusion
This project showcases the application of CNNs in image classification tasks, demonstrating their ability to handle complex datasets like Fashion MNIST. By using deep learning, the model achieved significant accuracy in distinguishing various fashion items, highlighting CNNs' effectiveness in visual recognition tasks.

## Future Work 
- Explore transfer learning with pre-trained models to compare results.

## Acknowledgements
- Fashion MNIST dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- TensorFlow and Keras: Deep learning libraries for building neural networks.

## Author
Ondrej Hruby
