import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss
import flask
from flask import Flask, request, render_template
import os
# Load the PyTorch FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the LFW dataset(replace it with your own dataset)
dataset_path = r'C:\Users\DELL\OneDrive\Desktop\Masterclass project\lfw'
images = []
labels = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            faces = mtcnn(img)

            if faces is None:
                continue  # Skip if no faces are detected

            for face in faces:
                face = face.permute(1, 2, 0).detach().cpu().numpy()  # Detach, move to CPU, and convert to NumPy array (HWC format)
                face = cv2.resize(face, (160, 160))                 # Resize using OpenCV
                face = transforms.ToTensor()(face)                  # Convert back to tensor
                face = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(face)  # Normalize
                face = face.unsqueeze(0).to(device)                 # Add batch dimension and move to device


                # Get embedding using ResNet
                embedding = resnet(face)[0].detach().cpu().numpy()
                images.append(embedding)
                labels.append(os.path.basename(root))
# Create a FAISS index for efficient similarity search
if images:
    index = faiss.IndexFlatL2(len(images[0]))
    for embedding in images:
        index.add(np.array([embedding]))
# Create a mapping of celebrity names to image indices
celebrity_mapping = {}
for i, label in enumerate(labels):
    if label not in celebrity_mapping:
        celebrity_mapping[label] = []
    celebrity_mapping[label].append(i)

# Flask application for image upload and recognition
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        img = cv2.imread(uploaded_file.stream, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        faces, _ = mtcnn(img)

        for face in faces:
            face = face.astype(np.uint8)
            face = cv2.resize(face, (160, 160))
            face = transforms.ToTensor()(face)
            face = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(face)
            face = face.unsqueeze(0).to(device)

            # Get embedding using ResNet
            embedding = resnet(face)[0].cpu().numpy()

            D, I = index.search(np.array([embedding]), 10)  # Search for top 10 matches
            closest_matches = []
            for i in I[0]:
                label = labels[i]
                indices = celebrity_mapping[label]
                for index in indices:
                    if index not in closest_matches:
                        closest_matches.append(index)
                        break

            # Return a list of closest matches
            return render_template('matches.html', matches=[labels[i] for i in closest_matches])

    else:
        return 'No file selected'

if __name__ == '__main__':
    app.run(debug=True)

