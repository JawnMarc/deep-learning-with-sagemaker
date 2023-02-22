import json, sys, os, io
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms


def net():
    model = models.resnet18(weights="DEFAULT")
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    input_feat = model.fc.in_features
    model.fc = nn.Sequential(
            nn.Linear(input_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.65),
            nn.Linear(256, 133))        
    return model


def model_fn(model_dir):
    model = net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as mfile:
        model.load_state_dict(torch.load(mfile,  map_location=device))
              
    model.to(device).eval()
              
    return model


# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, request_content_type):
    '''
    Takes request data and deserializes the data into an object for prediction
    '''
    assert request_content_type == 'images/jpeg'
    image = Image.open(io.BytesIO(request_body))
              
    trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return trans(image)

    
# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    '''
    Takes the deserialized request object and performs inference against the loaded model.
    '''
    # model.eval()
    with torch.no_grad():
        pred = model(input_object.unsqueeze(0))
    return pred



# Serialize the prediction result into the desired response content type
def output_fn(prediction, response_content_type):
    '''
    Takes the result of prediction and serializes this according to the response content type.
    '''
    assert response_content_type == 'application/json'
    respone = prediction.cpu().numpy().tolist()
        return json.dumps(response)