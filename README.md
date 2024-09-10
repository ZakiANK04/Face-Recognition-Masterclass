# Face Recognition App in Python

This Markdown file provides an overviThis Markdown document provides an overview of a face recognition app implemented in Python. The app utilizes the FaceNet model for face embedding and similarity search. Below, we’ll cover the main components and steps involved:
## Prerequisites

Before running the app, make sure you have the following installed:

- Python (3.6 or later)
- OpenCV (`pip install opencv-python`)
- PyTorch (`pip install torch`)
- `facenet-pytorch` library (`pip install facenet-pytorch`)
- `faiss` library (`pip install faiss-cpu`)
- Flask (`pip install flask`)

## Workflow
### 1-Loading the FaceNet Model:We load the FaceNet model, which consists of two parts:
 - MTCNN (Multi-task Cascaded Convolutional Networks): Used for face detection.
 - InceptionResnetV1: Used for face embedding (generating a compact representation of each face).
### 2- Dataset Preparation:
 - Replace the 'dataset_path' variable with the actual path to your dataset (e.g., LFW dataset).
 - The app assumes that your dataset contains face images organized in subdirectories (one subdirectory per person
### 3- Face detection and embedding : for each image in the dataset
 - Detect faces using MTCNN.
 - Resize the detected face to a consistent size (e.g., 160x160 pixels).
 - Normalize the face image.
 - Compute the face embedding using InceptionResnetV1.
 - Store the embeddings and corresponding labels (person names).
### 4- Creating a FAISS index:
 - If there are valid face embeddings, create a FAISS index for efficient similarity search.
 - The index is based on the L2 distance metric
### 5- Flask web app :
 - We create a simple Flask web application for image upload and recognition.
 - The app has two routes:
 - /: Displays an upload form.
 - /upload: Handles uploaded images.
 - When an image is uploaded:
 - Detect faces in the image.
 - Compute embeddings for each detected face.
 - Search the FAISS index for the closest matches (top 10).
 - Return the names of the closest-matching celebrities.
## Usage:
 1. Run the Flask app (python app.py).
 2. Access the app in your web browser (usually at http://localhost:5000).
 3.  Upload an image containing faces.
 4.  Receive the names of the closest-matching celebrities.



