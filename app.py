import streamlit as st
import pickle
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
        menu = ["Home","Machine Learning Basics","Image Training","Neural Network","Convolutional Neural Network"]
        app_mode = st.sidebar.selectbox("Menu",menu)

        
        

        # Home page
        if app_mode == "Home":
            st.title("ML and NLP Interactive Learning Platform")
            st.write("Welcome to our interactive platform to learn machine learning concepts.")
            app_mode_home = st.sidebar.selectbox('Radio',[1, 2])
            with st.columns([1, 10, 1])[1]:
                #st.image('', use_column_width=True, caption="Animated GIF on Home Page")
                add_vertical_space(2)
            

        if app_mode == "Machine Learning Basics":
            st.title("Machine Learning Basics")
                    
                # Add content for the machine learning basics page
            st.write("In this section, you will learn the fundamentals of machine learning.")
                    
            add_vertical_space(2)

                # You can add explanations, interactive widgets, and exercises here 
            st.write("Machine learning is a subfield of artificial intelligence that focuses on creating algorithms and models that enable computers to learn from data. It involves building systems that can automatically improve their performance on a specific task with experience, without being explicitly programmed. Machine learning is used in a wide range of applications, including image recognition, natural language processing, recommendation systems, and more.")
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

        

        if app_mode == "Neural Network":
                                                                                    
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


        if app_mode == "Convolutional Neural Network":

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