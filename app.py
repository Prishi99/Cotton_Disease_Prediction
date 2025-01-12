import sys
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.autograd import Variable
from werkzeug.utils import secure_filename
from PIL import Image
from flask import Flask, request, render_template

# Create required directories
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

data_dir = "data"
if not os.path.exists(os.path.join(data_dir, 'train')):
    os.makedirs(os.path.join(data_dir, 'train'))

# Constants
image_size = 32
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Image transformation
loader = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
])

# Try to load dataset, if not available use default classes
try:
    dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=ToTensor())
except:
    print("Warning: Could not load dataset from data/train. Using default classes.")
    class DummyDataset:
        def __init__(self):
            self.classes = ['disease_1', 'disease_2', 'disease_3', 'healthy']
    dataset = DummyDataset()

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(), 
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        
    def forward(self, xb):
        return self.network(xb)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    """Predict image class"""
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), get_default_device())
    # Get predictions from model
    with torch.no_grad():
        yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

def image_loader(image_path):
    """Load image and return appropriate tensor"""
    device = get_default_device()
    try:
        image = Image.open(image_path).convert('RGB')
        image = loader(image).float()
        return image.to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
try:
    device = get_default_device()
    model = CnnModel()
    
    # Load the model with appropriate device handling
    if device.type == 'cuda':
        model.load_state_dict(torch.load('modelCottonDemo.pth'))
    else:
        model.load_state_dict(torch.load('modelCottonDemo.pth', map_location=torch.device('cpu')))
    
    model = to_device(model, device)
    model.eval()
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            if model is None:
                return "Model not loaded properly", 500

            # Get the file from post request
            f = request.files['file']
            if not f:
                return "No file uploaded", 400

            # Save the file
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, UPLOAD_FOLDER, secure_filename(f.filename))
            f.save(file_path)

            # Load and predict
            image = image_loader(file_path)
            if image is None:
                return "Error loading image", 400

            pred = predict_image(image, model)

            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

            return pred

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error processing request", 500

    return None

if __name__ == '__main__':
    app.run(debug=True)