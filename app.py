import streamlit as st
import pickle
import torch
import seaborn as sns
import time
import torch.nn as nn
import torch.nn.functional as F 
from model_nn import ClassifierModule
from model_cnn import Cnn
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image, ImageFilter

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_drawable_canvas import st_canvas
from torchvision import datasets
import torch.optim as optim
from cnn_digit_recogniser import CNNDigitRecognizer

# Main Application
def main():
        # Create the app
        #st.title('Vizuara AI Labs - Handwritten Text Classification')
        st.sidebar.title('Navigation 😀 ')
        menu = ["Home 🏠","Machine Learning Basics 📖", "Dataset 📚", "Neural Networks 🧠", "Neural Networks and MNIST 🔢", "Convolutional Neural Networks 🤔", "CNN and MNIST 🧩"]
        app_mode = st.sidebar.selectbox("Menu",menu)

        # Function to sharpen an image
        # def sharpen_image(image):
        #     kernel = np.array([[-1, -1, -1],
        #                     [-1, 9, -1],
        #                     [-1, -1, -1]])
        #     sharpened = cv.filter2D(image, -1, kernel)
        #     return sharpened
        
        def add_vertical_space(space):
            for _ in range(space):
                st.write("")

        # Load dataset with caching
        #@st.cache_data(hash_funcs={torch.Tensor: lambda x: x.tolist()})
        # def load_mnist_dataset():
        #     mnist = fetch_openml('mnist_784', as_frame=False, cache=False, version=1)
        #     X = mnist.data.astype('float32') / 255.0
        #     y = mnist.target.astype('int64')
        #     return X, y
        

        # Function to preprocess image for inference
        def preprocess_image(image_path):
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            transform = transforms.Compose([transforms.Resize((28, 28)),
                                            transforms.ToTensor()])
            image = transform(image)
            return image

        def plot_example(X, y):
                    """Plot the first 25 images in a 5x5 grid."""
                    plt.figure(figsize=(10, 10))  # Set figure size to be larger (you can adjust as needed)

                    for i in range(5):  # For 5 rows
                        for j in range(5):  # For 5 columns
                            index = i * 5 + j
                            plt.subplot(5, 5, index + 1)  # 5 rows, 5 columns, current index
                            plt.imshow(X[index].reshape(28, 28), cmap='gray')  # Display the image in grayscale
                            plt.xticks([])  # Remove x-ticks
                            plt.yticks([])  # Remove y-ticks
                            plt.title(y[index], fontsize=8)  # Display the label as title with reduced font size

                    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
                    plt.tight_layout()  # Adjust the spacing between plots for better visualization
                    st.pyplot()  # Display the entire grid

        # Home page
        if app_mode == "Home 🏠":
            st.title("ML and NLP Interactive Learning Platform")
            add_vertical_space(2)
            st.header("Welcome to our interactive platform to learn machine learning concepts.")
            add_vertical_space(3)
            # app_mode_home = st.sidebar.selectbox('Radio',[1, 2])
            with st.columns([1, 10, 1])[1]:
                st.image('images/OyGx.gif')
                add_vertical_space(2)
            st.write("Let us explore what we can learn today. Go to the navigation bar and start learning!!")
         
        # ML Basics Page
        if app_mode == "Machine Learning Basics 📖":
            st.title("Machine Learning !!!")
                    
                # Add content for the machine learning basics page
            st.write("In this section, you will learn the fundamentals of machine learning.")
                    
            add_vertical_space(2)

                # You can add explanations, interactive widgets, and exercises here 
            st.write("Machine learning is a subfield of artificial intelligence that focuses on creating algorithms and models that enable computers to learn from data. It involves building systems that can automatically improve their performance on a specific task with experience, without being explicitly programmed.")
            st.write(" Machine learning is used in a wide range of applications, including image recognition, natural language processing, recommendation systems, and more.")
            with st.columns([1, 5, 1])[1]:
                
                st.image ('images/ML_basics.png')
                #sharpened_image = sharpen_image(image)
                
                # #image('images/ML_basics.png', use_column_width=True)
                # st.image(sharpened_image)
                add_vertical_space(2)

            st.write("Key concepts in machine learning include data preprocessing, model training, and evaluation. Common algorithms include linear regression, decision trees, and neural networks.")
            st.write("There are three key aspects of Machine Learning which are as follows: ") 
            aspects = [
                        "<b>Task</b>: A task is defined as the main problem in which we are interested. This task/problem can be related to the predictions and recommendations and estimations, etc.",
                        "<b>Experience</b>: It is defined as learning from historical or past data and used to estimate and resolve future tasks.",
                        "<b>Performance</b>: It is defined as the capacity of any machine to resolve any machine learning task or problem and provide the best outcome for the same. However, performance is dependent on the type of machine learning problems."]
            for aspect in aspects:
                st.markdown(f"- {aspect}", unsafe_allow_html=True)

        # MNIST Dataset
        if app_mode == "Dataset 📚":
            # Page Title
            st.title("Understanding Datasets in Neural Networks")

            # Introduction
            st.write("Welcome to the world of neural networks and machine learning! In this adventure, we'll explore the role of datasets in training machines to recognize and understand the world.")

            # What is a Dataset?
            st.subheader("What is a Dataset?")
            st.write("In the world of machine learning and neural networks, a dataset is like a treasure chest of information. It's a collection of data that's used to teach machines about the world.")

            # Task, Experience, Performance
            st.subheader("Three Key Aspects")
            st.write("There are three key aspects of using datasets in machine learning:")
            st.markdown("- **Task**: A task is the main problem we want the machine to solve. It can be anything from recognizing handwritten digits to understanding spoken language.")
            st.markdown("- **Experience**: Experience comes from historical or past data. Machines learn from this data to get better at solving tasks in the future.")
            st.markdown("- **Performance**: This is how well the machine can solve a task. The more data and experience it has, the better its performance!")

            # Image
            st.image("images/datasets.png", caption="Datasets are Like Treasure Chests", use_column_width=True)

            # # Interactive Example
            # st.subheader("Imagine You Have a Dataset")
            # st.write("Think of a dataset like a giant collection of pictures, stories, and information. Let's pretend you have a dataset about different animals.")

            # st.image("images/animals.png", caption="Your Animal Dataset", use_column_width=True)
            
            # Function to recognize a drawn digit
            def recognize_digit(digit_image, model):
                print(digit_image)
                digit_image = Image.fromarray(digit_image)
                print(digit_image)
                transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28)), transforms.ToTensor()])
                digit_image = transform(digit_image).unsqueeze(0)
                with torch.no_grad():
                    output = model(digit_image)
                predicted_digit = output.argmax(dim=1).item()
                return predicted_digit

            # Define the pages and subpages
            pages = ["Intro", "Explore MNIST", "Draw a Digit", "Preprocessing"]
            subpages = ["Blurring", "RGB Color Change"]

            # Sidebar navigation
            st.sidebar.title("Navigate")
            page = st.sidebar.selectbox("Go to", pages)

            # Main content
            st.title("MNIST Dataset Exploration")

            # Load a pre-trained model
            model = CNNDigitRecognizer()
            #model.load_state_dict(torch.load('cnn_digit_recognizer.pth'))
            model.eval()

            if page == "Intro":
                st.write("Welcome to the world of MNIST!")
                st.write("In this journey, we'll explore the fascinating MNIST dataset, learn how to draw digits, and understand data preprocessing techniques.")
                st.write("Let's start our adventure!Go to the sidebar and Explore MNIST")

            if page == "Explore MNIST":
                st.write("This is the page for exploring MNIST.")

                # Add content for exploring MNIST
                st.title("MNIST Dataset")
                st.write("The MNIST dataset is a collection of handwritten digits widely used in machine learning.")
                
                add_vertical_space(2)
                st.write("It contains 28x28 grayscale images of digits from 0 to 9. Let's explore some sample images:")
                st.image("images/MNIST-sample.png")

                # Sample MNIST images
                #mnist_images = [Image.open(f"images/mnist_{i}.png") for i in range(10)]
                #st.image(mnist_images, caption=[str(i) for i in range(10)], width=100)

                st.subheader("What Is MNIST?")
                st.write("MNIST stands for Modified National Institute of Standards and Technology.")
                st.write("It's a dataset of handwritten digits used to train and test various machine learning models.")
                with st.columns([1, 10, 1])[1]:
                    st.image("images/mnist-3.png")
                st.write("Each image is 28x28 pixels, and the dataset contains 60,000 training and 10,000 test images.")

                st.subheader("Why Is MNIST Important?")
                st.write("MNIST is a fundamental dataset in the machine learning world.")
                st.write("It's often used as a benchmark to test new models and algorithms.")
                st.write("Many beginners start their journey by working with MNIST to learn image classification.")

            if page == "Draw a Digit":
                st.write("This is the page for drawing a digit.")
                # Add content for drawing a digit
                st.title("Try Drawing a Digit")
                st.write("You can draw a digit here and see how machine learning models recognize it.")
                st.write("Draw a digit and click the 'Recognize' button to see the prediction.")

                # Create a canvas for drawing a digit
                canvas = st_canvas(
                    fill_color="black",
                    stroke_width=20,
                    stroke_color="white",
                    width=200,
                    height=200,
                    drawing_mode="freedraw",
                )

                # Button to recognize the drawn digit
                if st.button("Recognize"):
                    st.write("Machine learning model recognizes the drawn digit:")
                    recognized_digit = recognize_digit(canvas.image_data, model)
                    st.write(f"Recognized Digit: {recognized_digit}")

            if page == "Preprocessing":
                st.write("This is the page for data preprocessing.")
                st.title("Preprocessing Techniques for MNIST")
                st.write("Before feeding images to a machine learning model, we often apply preprocessing techniques to enhance the data.")
                st.write("Let's explore two common preprocessing techniques: blurring and RGB color change.")
                subpage = st.radio("Select Preprocessing Technique", subpages)
                
                # Subpage navigation buttons
                if st.button("Blur Images"):
                    subpage = "Blurring"
                elif st.button("Change RGB Color"):
                    subpage = "RGB Color Change"
                else:
                    subpage = "Preprocessing"

                # Add a button to go back to the preprocessing page
                if subpage != "Preprocessing":
                    if st.button("Back to Preprocessing"):
                        subpage = "Preprocessing"

                # Blur Images
                if subpage == "Blurring":
                    st.subheader("Blurring Images")
                    st.write("Blurring reduces noise and can make it easier for the machine to focus on the important features of an image.")
                    st.write("Here's an example of a blurred MNIST digit:")
                    st.image("images/blurred_mnist.png", caption="Blurred MNIST Digit")

                    # Add explanation and code for blurring
                    st.subheader("How to Apply Blurring")
                    st.write("You can apply blurring using various methods, such as Gaussian blur. Here's a code snippet to get you started:")

                    st.code(
                        """
                        # Load an image
                        image = Image.open('image.jpg')

                        # Apply Gaussian blur
                        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
                        """
                    )

                # Change RGB Color
                if subpage == "RGB Color Change":
                    st.subheader("Changing RGB Color")
                    st.write("You can change the RGB color of an image by modifying its color channels.")
                    st.write("Here's a code snippet to get you started with changing the color of an image:")

                    st.code(
                        """
                        # Load an image
                        image = Image.open('image.jpg')

                        # Separate RGB channels
                        red_channel, green_channel, blue_channel = image.split()

                        # Modify the channels (for example, swapping red and blue channels)
                        modified_image = Image.merge("RGB", (blue_channel, green_channel, red_channel))
                        """
                    )

            # Button to navigate back to the introduction
            if page != "Intro":
                if st.button("Go to navigation"):
                    page = "Intro"
        
        # NN page
        if app_mode == "Neural Networks 🧠":
            st.title("Let's explore Neural Networks !!")
            add_vertical_space(2)
            # Introduction to neural networks
            st.write("Neural networks are like virtual brains that help computers learn and make decisions, just like how you learn from experiences. They are designed to mimic the functioning of the human brain, consisting of interconnected layers of neurons. Neural networks are a fundamental concept in machine learning and deep learning. Neural networks are used for various tasks, including image recognition, natural language processing, and more. ")
            add_vertical_space(2)

            # Brain analogy
            st.write("Imagine a neural network as a group of connected cells. Each cell can understand something different, like shapes or colors. When we show the network a picture, these cells work together to recognize what's in the picture.")
            # Feedforward neural network diagram
            st.image("images/feedforward_neural_network.png", caption="Feedforward Neural Network")

            # Layers and neurons
            st.write("Neural networks are organized in layers, and each layer contains many cells, like in our brain. There are three types of layers:")
            st.markdown("- Input Layer: It takes information from the outside world, like a picture.")
            st.markdown("- Hidden Layers: These layers think and make decisions based on the input.")
            st.markdown("- Output Layer: It gives the final answer, like 'It's a cat!'")
            
            add_vertical_space(2)
            # Activation functions
            st.write("Activation functions help cells in the network to decide how much they should 'fire.' The most common activation functions are like on-off switches.")
            st.markdown("- ReLU: It's like turning on a light switch when things are bright enough.")
            st.markdown("- Sigmoid: It's like a dimmer switch that can be turned up or down.")
            st.image("images/sigmoid.png")
            st.markdown("- Tanh: It's like a thermometer showing how hot or cold something is.")

            add_vertical_space(2)
            # Loss functions
            st.write("Loss functions help the network know how well it's doing. If it makes a mistake, the loss goes up. The network tries to lower the loss. Two common loss functions are:")
            st.markdown("- Mean Squared Error: It checks how far the network's answer is from the right answer.")
            st.markdown("- Cross-Entropy Loss: It's like counting how surprised the network is about the answer.")

            add_vertical_space(2)
            
            # Training and backpropagation
            st.image("images/backpropogation-nn.png")
            st.write("Neural networks learn from their mistakes. They keep trying to make fewer mistakes. This is called training. When they make a mistake, they use backpropagation to figure out how to fix it.")
            
            add_vertical_space(2)

            st.write("Here's a simple example of a feedforward neural network in code:")
            st.code("""
            import torch
            import torch.nn as nn

            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init()
                    self.fc1 = nn.Linear(in_features=784, out_features=128)
                    self.fc2 = nn.Linear(in_features=128, out_features=10)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
            """, language="python")

            # Interactive feedforward neural network example
            st.subheader("Interactive Feedforward Neural Network Example")

            # Allow kids to adjust the number of hidden neurons
            num_hidden_neurons = st.slider("Number of Hidden Neurons", min_value=1, max_value=256, value=64)

            # Create a simple feedforward neural network
            class NeuralNetwork(nn.Module):
                def __init__(self, input_dim=784, hidden_dim=num_hidden_neurons, output_dim=10):
                    super(NeuralNetwork, self).__init()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, output_dim)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)
                    return x
                
            # Explanation
            st.markdown("You can adjust the number of hidden neurons in the network. Think of these neurons as tiny helpers that work together to understand things. More neurons can help the network learn complex stuff.")

            # Create a simple image to visually represent the neural network
            image_url = "images/mlp_mnist.png" 
            
            st.write("Let's explore the neural network:")
            st.write(f"- It has an Input Layer with 784 neurons, which is like the network's eyes.")
            st.write(f"- You can adjust the Hidden Layer to have {num_hidden_neurons} neurons. It's like the network's brain.")
            st.write(f"- The Output Layer has 10 neurons, which is like the network's mouth.")
            
            st.image(image_url, caption="Neural Network Diagram")

            # Interactive description
            st.write("Imagine this network is learning to recognize handwritten numbers. When you adjust the number of hidden neurons, you're making its brain more or less complex.")
            st.write(f"With {num_hidden_neurons} neurons, it's getting smarter and can recognize more details in the numbers.")

            # impact of the network
            if num_hidden_neurons >= 128:
                st.write("Look how well it's doing with many neurons! It's like a super-smart computer.")
            else:
                st.write("With fewer neurons, it's still learning, but it might make some mistakes. Just like when you're learning something new!")

            st.image("images/mnist_layer.png")
            # Explanation
            st.markdown("The image above shows how the network looks. It has an input layer (receiving data), a hidden layer (thinking and making decisions), and an output layer (giving the final answer). The hidden layer can have as many neurons as you decide.")

        # NN model
        if app_mode == "Neural Networks and MNIST 🔢":
            st.sidebar.subheader("Navigation")
            nn_menu = ["Load Dataset", "Training Parameters", "Result Explanation", "Inference"]
            nn_app_mode = st.sidebar.selectbox("Menu", nn_menu, key="nn_menu")

            # Initialize variables
            mnist_loaded = False
            X_train, X_test, y_train, y_test = None, None, None, None
            net = None

            if 'nn_app_mode' not in st.session_state:
                st.session_state.nn_app_mode = "Load Dataset"

            if st.session_state.nn_app_mode == "Load Dataset":
                st.title("Neural Networks and MNIST Image Dataset")
                st.write("Welcome to the Neural Networks and MNIST Image Dataset app. Let's get started by loading the dataset.")

                mnist = fetch_openml('mnist_784', as_frame=False, cache=False, version=1)
                X = mnist.data.astype('float32') / 255.0
                y = mnist.target.astype('int64')

                # Split dataset
                st.write("Splitting dataset")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                st.session_state.mnist_loaded = True

                # # Show a 5x5 grid of loaded MNIST images
                # st.subheader('Sample Training Images and Labels')
                # st.write('Here are some example images from the MNIST dataset')

                # plot_example(X_train, y_train)

                # Add Training Parameters section on the same page
                st.title("Neural Networks and MNIST Image Dataset")
                st.write("Let's configure the training parameters.")

                st.write("In this section, you can choose training parameters for your neural network model.")
                st.header("Training Parameters")

                epochs = st.slider("Epochs", min_value=1, max_value=20, step=1)
                learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, step=0.01)

                st.write("You can adjust the number of training epochs and the learning rate to influence model training.")

                if st.button("Train model"):
                    with st.spinner("Training model..."):
                        # Train the model
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'

                        mnist_dim = X_train.shape[1]
                        hidden_dim = int(mnist_dim / 8)
                        output_dim = len(set(y_train))

                        net = NeuralNetClassifier(ClassifierModule(mnist_dim, hidden_dim, output_dim))
                        criterion = torch.nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                        device = device

                        for epoch in range(epochs):
                            net.train()
                            optimizer.zero_grad()
                            outputs = net(torch.from_numpy(X_train))
                            loss = criterion(outputs, torch.from_numpy(y_train))
                            loss.backward()
                            optimizer.step()

                        st.session_state.net = net

            elif st.session_state.nn_app_mode == "Result Explanation":
                st.title("Result Explanation")
                st.write("In this section, you will explore the results of the model training based on the chosen parameters.")

                if st.session_state.net is not None:
                    if st.button("Show Results", key="show_results_btn"):
                        net.eval()
                        net.fit(X_train, y_train)
                        y_pred = net.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f'Accuracy: {accuracy:.3%}')

                        # Confusion Matrix as an example of result visualization
                        cm = confusion_matrix(y_test, y_pred)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title('Confusion Matrix')
                        st.pyplot()

                        # Show a 5 x 5 grid of MNIST images that were misclassified
                        misclassified_indices = np.where(y_pred != y_test)[0][:25]
                        st.write("Misclassified Images:")
                        for idx in misclassified_indices:
                            st.image(X_test[idx].reshape(28, 28), caption=f"True: {y_test[idx]}, Predicted: {y_pred[idx]}")

            elif st.session_state.nn_app_mode == "Inference":
                st.title("Inference")
                st.write("In this section, you can upload an image for model inference.")

                uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

                if uploaded_image is not None:
                    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

                    # Perform inference on the uploaded image using the trained model
                    if st.session_state.net is not None:
                        st.write("Performing Inference:")
                        image_tensor = preprocess_image(uploaded_image)
                        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                        st.session_state.net.eval()
                        with torch.no_grad():
                            output = st.session_state.net(image_tensor)
                        predicted_class = torch.argmax(output).item()
                        st.write(f"Predicted Class: {predicted_class}")

        # CNN page
        if app_mode == "Convolutional Neural Networks 🤔":
            st.title("Let's Explore Convolutional Neural Networks (CNNs)")

            # Introduction to CNNs
            st.write("Convolutional Neural Networks (CNNs) are like superheroes for understanding images and patterns. They can spot objects in photos and videos, just like how you find hidden treasures!")

            # Superhero analogy
            st.write("Think of a CNN as a superhero with special glasses. These glasses help them see tiny details in pictures. They're amazing at finding clues!")

            # Layers and filters
            st.write("A CNN is made up of special layers and filters. Let's uncover their powers:")
            st.markdown("- **Convolutional Layer**: These are like the superhero's magnifying glasses. They zoom in on small parts of a picture to uncover secrets.")
            st.markdown("- **Pooling Layer**: Imagine the superhero jotting down important notes from the clues they find.")
            st.markdown("- **Fully Connected Layer**: It's like the superhero putting all the pieces of the puzzle together to solve the mystery.")

             # Interactive example
            st.video("https://www.youtube.com/watch?v=K_BHmztRTpA")

        # CNN model                 
        if app_mode == "CNN and MNIST 🧩":
            st.sidebar.subheader("Navigation")
            cnn_menu = ["Load Dataset", "Training Parameters", "Result Explanation", "Inference"]
            cnn_app_mode = st.sidebar.selectbox("Menu", cnn_menu, key="cnn_menu")

            # Initialize variables
            mnist_loaded = False
            XCnn_train, XCnn_test, y_train, y_test = None, None, None, None
            cnn = None

            if 'cnn_app_mode' not in st.session_state:
                st.session_state.cnn_app_mode = "Load Dataset"

            if st.session_state.cnn_app_mode == "Load Dataset":
                st.title("Convolutional Neural Network and MNIST Image Dataset")
                st.write("Welcome to the Convolutional Neural Network and MNIST Image Dataset app. Let's get started by loading the dataset.")

                mnist = fetch_openml('mnist_784', as_frame=False, cache=False, version=1)
                X = mnist.data.astype('float32')
                y = mnist.target.astype('int64')
                X /= 255.0

                XCnn = X.reshape(-1, 1, 28, 28)
                st.write("Splitting dataset")
                XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

                st.session_state.mnist_loaded = True

                # # Show a 5x5 grid of loaded MNIST images
                # st.subheader('Sample Training Images and Labels')
                # st.write('Here are some example images from the MNIST dataset')

                # plot_example(XCnn_train.squeeze(), y_train)

                # Add Training Parameters section on the same page
                st.title("Convolutional Neural Network and MNIST Image Dataset")
                st.write("Let's configure the training parameters.")

                st.write("In this section, you can choose training parameters for your convolutional neural network model.")
                st.header("Training Parameters")

                epochs = st.slider("Epochs", min_value=1, max_value=20, step=1)
                learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, step=0.01)

                st.write("You can adjust the number of training epochs and the learning rate to influence model training.")

                if st.button("Train model"):
                    with st.spinner("Training model..."):
                        # Train the model
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'

                        cnn = NeuralNetClassifier(Cnn).to(device)
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
                        device = device

                        for epoch in range(epochs):
                            cnn.train()
                            optimizer.zero_grad()
                            outputs = cnn(torch.from_numpy(XCnn_train).to(device))
                            loss = criterion(outputs, torch.from_numpy(y_train).to(device))
                            loss.backward()
                            optimizer.step()

                        st.session_state.cnn = cnn

            elif st.session_state.cnn_app_mode == "Result Explanation":
                st.title("Result Explanation")
                st.write("In this section, you will explore the results of the model training based on the chosen parameters.")

                if st.session_state.cnn is not None:
                    if st.button("Show Results", key="show_results_btn"):
                        cnn.eval()
                        cnn.fit(XCnn_train, y_train)
                        #XCnn_test_tensor = torch.from_numpy(XCnn_test).to(device)
                        y_pred_cnn = cnn.predict(XCnn_test)
                        accuracy = accuracy_score(y_test, y_pred_cnn)

                        st.write(f'Accuracy: {accuracy:.3%}')

                        # Confusion Matrix as an example of result visualization
                        cm = confusion_matrix(y_test, y_pred_cnn)
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
                        plt.xlabel('Predicted')
                        plt.ylabel('True')
                        plt.title('Confusion Matrix')
                        st.pyplot()

                        # Show a 5 x 5 grid of MNIST images that were misclassified
                        misclassified_indices = np.where(y_pred_cnn != y_test)[0][:25]
                        st.write("Misclassified Images:")
                        for idx in misclassified_indices:
                            st.image(XCnn_test[idx].squeeze(), caption=f"True: {y_test[idx]}, Predicted: {y_pred_cnn[idx]}")

            elif st.session_state.cnn_app_mode == "Inference":
                st.title("Inference")
                st.write("In this section, you can upload an image for model inference.")

                uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

                if uploaded_image is not None:
                    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

                    # Perform inference on the uploaded image using the trained model
                    if st.session_state.cnn is not None:
                        st.write("Performing Inference:")
                        image_tensor = preprocess_image(uploaded_image)
                        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                        st.session_state.cnn.eval()
                        with torch.no_grad():
                            output = st.session_state.cnn(image_tensor.to(device))
                        predicted_class = torch.argmax(output).item()
                        st.write(f"Predicted Class: {predicted_class}")
      
     
if __name__ == '__main__':
    st.set_page_config(page_title="Handwritten Text Classification", layout="wide") 
    main()
