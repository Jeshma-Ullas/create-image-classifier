import torch
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

def create_network(structure='vgg16', dropout=0.1, hidden_units=4096, lr=0.001, device='cpu'):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
    model.to(device)
    
    return model, criterion, optimizer

def save_model_checkpoint(train_data, model, path='checkpoint.pth', structure='vgg16', hidden_units=4096, dropout=0.3, lr=0.001, epochs=1):
    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'structure': structure,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': lr,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, path)

def load_saved_model(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    lr = checkpoint['learning_rate']
    
    model, criterion, optimizer = create_network(structure, dropout, hidden_units, lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, criterion, optimizer

def perform_prediction(image_path, model, topk=5, device='cpu'):
    device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
    model.to(device)
    model.eval()
    
    image = process_image(image_path).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    probs = torch.exp(output).cpu().numpy()[0]
    top_idx = np.argsort(-probs)[:topk]
    top_classes = [str(idx) for idx in top_idx]
    top_probabilities = probs[top_idx]
    
    return top_probabilities, top_classes

def preprocess_image(image):
    img = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    
    return img

def load_category_names_from_file(json_file='cat_to_name.json'):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
