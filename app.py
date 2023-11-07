import streamlit as st
import pickle
import cv2
import torch
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
import requests
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_drawable_canvas import st_canvas
import base64
import io
from bokeh.models import Div
from bokeh.plotting import figure
from bokeh.models import Button
from bokeh.layouts import layout
from torchvision import datasets
import torch.optim as optim
from torch.utils.data import DataLoader






# Preprocessng data 
#mnist = fetch_openml('mnist_784', as_frame=False, cache=False, version=1)
#X = mnist.data.astype('float32')
#y = mnist.target.astype('int64')
#X /= 255.0


# Main Application
def main():
        # Create the app
        #st.title('Vizuara AI Labs - Handwritten Text Classification')
        st.sidebar.title('Navigation')
        menu = ["Home","Machine Learning Basics", "Neural Networks","Convolutional Neural Networks", "Dataset","Image Training", "Neural Networks model", "Convolutional Neural Network model"]
        app_mode = st.sidebar.selectbox("Menu",menu)

        
        

        # Home page
        if app_mode == "Home":
            st.title("ML and NLP Interactive Learning Platform")
            add_vertical_space(2)
            st.header("Welcome to our interactive platform to learn machine learning concepts.")
            add_vertical_space(3)
            app_mode_home = st.sidebar.selectbox('Radio',[1, 2])
            with st.columns([1, 10, 1])[1]:
                st.image('images/OyGx.gif')
                add_vertical_space(2)
            st.write("Let us explore what we can learn today. Go to the navigation bar and start learning!!")
            

        if app_mode == "Machine Learning Basics":
            st.title("Machine Learning !!!")
                    
                # Add content for the machine learning basics page
            st.write("In this section, you will learn the fundamentals of machine learning.")
                    
            add_vertical_space(2)

                # You can add explanations, interactive widgets, and exercises here 
            st.write("Machine learning is a subfield of artificial intelligence that focuses on creating algorithms and models that enable computers to learn from data. It involves building systems that can automatically improve their performance on a specific task with experience, without being explicitly programmed.")
            st.write(" Machine learning is used in a wide range of applications, including image recognition, natural language processing, recommendation systems, and more.")
            with st.columns([1, 5, 1])[1]:
                st.image('images/ML_basics.png', use_column_width=True)
                add_vertical_space(2)

            st.write("Key concepts in machine learning include data preprocessing, model training, and evaluation. Common algorithms include linear regression, decision trees, and neural networks.")
            st.write("There are three key aspects of Machine Learning which are as follows: ") 
            aspects = [
                        "<b>Task</b>: A task is defined as the main problem in which we are interested. This task/problem can be related to the predictions and recommendations and estimations, etc.",
                        "<b>Experience</b>: It is defined as learning from historical or past data and used to estimate and resolve future tasks.",
                        "<b>Performance</b>: It is defined as the capacity of any machine to resolve any machine learning task or problem and provide the best outcome for the same. However, performance is dependent on the type of machine learning problems."]
            for aspect in aspects:
                st.markdown(f"- {aspect}", unsafe_allow_html=True)
        
        if app_mode == "Neural Networks":
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
            image_url = "images/mlp_mnist.png"  # You can replace this with an actual image URL
            
            st.write("Let's explore the neural network:")
            st.write(f"- It has an Input Layer with 784 neurons, which is like the network's eyes.")
            st.write(f"- You can adjust the Hidden Layer to have {num_hidden_neurons} neurons. It's like the network's brain.")
            st.write(f"- The Output Layer has 10 neurons, which is like the network's mouth.")
            
            st.image(image_url, caption="Neural Network Diagram")

            # Interactive description
            st.write("Imagine this network is learning to recognize handwritten numbers. When you adjust the number of hidden neurons, you're making its brain more or less complex.")
            st.write(f"With {num_hidden_neurons} neurons, it's getting smarter and can recognize more details in the numbers.")

            # Show the impact of the network
            if num_hidden_neurons >= 128:
                st.write("Look how well it's doing with many neurons! It's like a super-smart computer.")
            else:
                st.write("With fewer neurons, it's still learning, but it might make some mistakes. Just like when you're learning something new!")

            st.image("images/mnist_layer.png")
            # Explanation
            st.markdown("The image above shows how the network looks. It has an input layer (receiving data), a hidden layer (thinking and making decisions), and an output layer (giving the final answer). The hidden layer can have as many neurons as you decide.")


        # CNN page
        if app_mode == "Convolutional Neural Networks":
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

            # # Interactive example
            # st.write("Welcome to te Superhero Adventure! Join our superhero, Super Detecto, on an exciting mission to uncover hidden patterns in an image. Get ready to be amazed!")
            # st.subheader("JAdjust Super Detecto's Powers")

            # # Allow kids to adjust the number of filters
            # num_filters = st.slider("Number of Filters", min_value=1, max_value=16, value=8)

            # # Show the impact of filters
            # st.write(f"With {num_filters} filters, our superhero can examine {num_filters} different parts of an image at once. They become super detectives!")

            # # Create an interactive adventure
            # if st.button("Start the Superhero Adventure"):
            #     st.write("Fantastic! You're about to embark on a superhero adventure with Super Detecto. Watch as Super Detecto uses their magic filters to reveal hidden patterns in the image. Get ready for some superhero action!")

            #     # Create a drawing canvas using Bokeh
            #     plot = figure(plot_width=400, plot_height=400, tools="pan,reset,save,box_zoom")
            #     plot.background_fill_color = "white"
            #     plot.outline_line_color = None

            #     # Add an image to the canvas
            #     img_url = "https://your-image-url.com"  # Replace with your adventure image URL
            #     plot.image_url(url=[img_url], x=0, y=0, w=1, h=1)

            #     # Define a callback function to process the canvas
            #     def process_canvas():
            #         # Implement the logic to process the canvas here
            #         st.write("Super Detecto is using their magic filters to uncover hidden patterns!")

            #     # Add a button to trigger Super Detecto's magic
            #     if st.button("Use Super Detecto's Magic"):
            #         process_canvas()

            #     # Display the Bokeh plot
            #     st.bokeh_chart(plot)

            # # Explain the adventure
            # if st.button("What's Happening?"):
            #     st.write("In this superhero adventure, Super Detecto scans the canvas to find special patterns. Each filter in their magic toolbox looks for something different. It's like a magical treasure hunt!")

            # # Add more interactive elements, explanations, and code snippets as needed
       
        if app_mode == "Dataset":
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

            # Interactive Example
            st.subheader("Imagine You Have a Dataset")
            st.write("Think of a dataset like a giant collection of pictures, stories, and information. Let's pretend you have a dataset about different animals.")

            st.image("images/animals.png", caption="Your Animal Dataset", use_column_width=True)
            
            
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28)), transforms.ToTensor()])

            # Create a custom dataset using this transform
            mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            
            # Define a simple feedforward neural network for digit recognition
            class DigitRecognizer(torch.nn.Module):
                def __init__(self):
                    super(DigitRecognizer, self).__init__()
                    self.fc1 = torch.nn.Linear(28 * 28, 128)
                    self.fc2 = torch.nn.Linear(128, 10)

                def forward(self, x):
                    x = x.view(-1, 28 * 28)
                    x = torch.relu(self.fc1(x))
                    x = self.fc2(x)
                    return torch.log_softmax(x, dim=1)

            model = DigitRecognizer()


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Load data into DataLoader
            train_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)



            # Training loop
            for epoch in range(5):
                for data, target in train_loader:
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()


            torch.save(model.state_dict(), 'digit_recognizer.pth')


            # Function to recognize a drawn digit
            def recognize_digit(digit_image, model):
                digit_image = Image.open(io.BytesIO(base64.b64decode(digit_image)))
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
            model = DigitRecognizer()
            model.load_state_dict(torch.load('digit_recognizer.pth'))
            model.eval()

            if page == "Intro":
                st.write("Welcome to the world of MNIST!")
                st.write("In this journey, we'll explore the fascinating MNIST dataset, learn how to draw digits, and understand data preprocessing techniques.")
                st.write("Let's start our adventure!")

            if page == "Explore MNIST":
                st.write("This is the page for exploring MNIST.")
                # Add content for exploring MNIST
                st.title("MNIST Dataset")
                st.write("The MNIST dataset is a collection of handwritten digits widely used in machine learning.")
                st.write("It contains 28x28 grayscale images of digits from 0 to 9. Let's explore some sample images:")

                # Sample MNIST images
                mnist_images = [Image.open(f"images/mnist_{i}.png") for i in range(10)]
                st.image(mnist_images, caption=[str(i) for i in range(10)], width=100)

                st.subheader("What Is MNIST?")
                st.write("MNIST stands for Modified National Institute of Standards and Technology.")
                st.write("It's a dataset of handwritten digits used to train and test various machine learning models.")
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
                    st.image("images/blurred_mnist.png", caption="Blurred MNIST Digit", use_column_width=True)

                    # Add explanation and code for blurring
                    st.subheader("How to Apply Blurring")
                    st.write("You can apply blurring using various methods, such as Gaussian blur. Here's a code snippet to get you started:")

                    st.code(
                        """
                        # Load an image
                        image = cv2.imread('image.jpg')

                        # Apply Gaussian blur
                        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
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
                        image = cv2.imread('image.jpg')

                        # Extract RGB channels
                        red_channel = image[:, :, 0]
                        green_channel = image[:, :, 1]
                        blue_channel = image[:, :, 2]

                        # Modify the channels (for example, swapping red and blue channels)
                        modified_image = cv2.merge([blue_channel, green_channel, red_channel])
                        """
                    )

            # Button to navigate back to the introduction
            if page != "Intro":
                if st.button("Back to Introduction"):
                    page = "Intro"



        if app_mode == "Image Training":
            # Print a selection of training images and their labels
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            st.subheader('Sample Training Images and Labels')
            st.write('Here are some example images from the MNIST dataset')

            # Add content to the "Loading Data" section
            st.header('Loading Data')
            
            # def plot_example(X, y):
            #     """Plot the first 100 images in a 10x10 grid."""
            #     plt.figure(figsize=(28, 28))  # Set figure size to be larger (you can adjust as needed)

            #     for i in range(10):  # For 10 rows
            #         for j in range(10):  # For 10 columns
            #             index = i * 10 + j
            #             plt.subplot(10, 10, index + 1)  # 10 rows, 10 columns, current index
            #             plt.imshow(X[index].reshape(28, 28))  # Display the image
            #             plt.xticks([])  # Remove x-ticks
            #             plt.yticks([])  # Remove y-ticks
            #             plt.title(y[index], fontsize=8)  # Display the label as title with reduced font size

            #     plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
            #     plt.tight_layout()  # Adjust the spacing between plots for better visualization
            #     #plt.show()  # Display the entire grid
            #     st.image

            # plot_example(X_train, y_train)

        

        if app_mode == "Neural Network model":
                                                                                    
            # Add content to the "Build Neural Network with Pytorch" section
            st.header("Build Neural Network with Pytorch")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            mnist_dim = X.shape[1]
            hidden_dim = int(mnist_dim/8)
            output_dim = len(np.unique(mnist.target))


            # Define a simple Pytorch model
            model = ClassifierModule(
                mnist_dim, hidden_dim, output_dim
            )

            # Create a button to train the model

            if st.button("Train Model"):
                net = NeuralNetClassifier(
                ClassifierModule,
                max_epochs=20,
                lr=0.1,
                device=device)
                
                net.fit(X_train, y_train)

                st.write("Training Complete")

            # Add content to the "Prediction" section
                st.header("Prediction")

                y_pred = net.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f'Accuracy: {accuracy:.3%}')
                #error_mask = y_pred != y_test
                #plot_example(X_test[error_mask], y_pred[error_mask])


        if app_mode == "Convolutional Neural Network model":

            # Add content to the "Convolutional Network" section
            st.header("Convolutional Network")

            XCnn = X.reshape(-1, 1, 28, 28)
            #XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

            # Interactive Features

            # Create a slider for the number of hidden units
            num_hidden_units = st.slider("Select the number of hidden units: ", min_value =1, max_value = 200, value = 98)

            # Create a button to train the model

            if st.button("Train Model"):
                cnn = NeuralNetClassifier(
                Cnn,
                max_epochs=10,
                lr=0.002,
                optimizer=torch.optim.Adam)

                cnn.fit(XCnn_train, y_train)

                st.write("Training Complete")


                st.header("Prediction")

                y_pred_cnn = cnn.predict(XCnn_test)

                accuracy = accuracy_score(y_test, y_pred_cnn)
                st.write(f'Accuracy: {accuracy:.3%}')
                #error_mask = y_pred != y_test
                #plot_example(X_test[error_mask], y_pred_cnn[error_mask])
           

                  

            # Hands-on Examples page
        if app_mode == "Hands-on Examples":
            st.title("Hands-on Examples")
                
                # Add content for the hands-on examples page
            st.write("Explore practical machine learning examples with interactive code snippets.")
                
                # You can add examples, code snippets, and interactive features here

            # Run the Streamlit app



if __name__ == '__main__':
    st.set_page_config(page_title="Handwritten Text Classification", layout="wide") 
    main()