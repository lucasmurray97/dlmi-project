import torch
import torch.nn as nn
from torch.autograd import Function
from utils_training import init_weights_xavier
from torchvision.models import resnet18, resnet34
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --------------------------
# Gradient Reversal Layer
# --------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)

# --------------------------
# Feature Extractor
# --------------------------

class FeatureExtractor(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            self.extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        else:
            self.extractor = backbone
        # Freeze the feature extractor
        for param in self.extractor.parameters():
            param.requires_grad = False
        # add three linear layers to the end of the network
        self.head = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 512))
        
    def forward(self, x):
        features = self.extractor(x)
        x = self.head(features)
        return x
        

# --------------------------
# Label Classifier
# --------------------------
class LabelClassifier(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim=256, num_classes=2, n_hidden_layers=2):
        super().__init__()
        self.classifier = nn.ModuleList()
        # Create the first layer
        self.classifier.append(nn.Linear(input_dim, hidden_dim))
        self.classifier.append(nn.ReLU())
        # Create the hidden layers
        for _ in range(n_hidden_layers):
            self.classifier.append(nn.Linear(hidden_dim, hidden_dim))
            self.classifier.append(nn.ReLU())
        # Create the output layer
        self.classifier.append(nn.Linear(hidden_dim, hidden_dim))
        self.classifier.append(nn.ReLU())
        self.classifier.append(nn.Linear(hidden_dim, num_classes))
        # Convert the list to a sequential module
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, features):
        return self.classifier(features)

# --------------------------
# Domain Discriminator
# --------------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim=256, num_dom=5, n_hidden_layers=2):
        super().__init__()
        self.discriminator = nn.ModuleList()
        # Create the first layer
        self.discriminator.append(nn.Linear(input_dim, hidden_dim))
        self.discriminator.append(nn.ReLU())
        # Create the hidden layers
        for _ in range(n_hidden_layers):
            self.discriminator.append(nn.Linear(hidden_dim, hidden_dim))
            self.discriminator.append(nn.ReLU())
        # Create the output layer
        self.discriminator.append(nn.Linear(hidden_dim, hidden_dim))
        self.discriminator.append(nn.ReLU())
        self.discriminator.append(nn.Linear(hidden_dim, num_dom))
        # Convert the list to a sequential module
        self.discriminator = nn.Sequential(*self.discriminator)

    def forward(self, features):
        return self.discriminator(features)

# --------------------------
# DANN Model Wrapper
# --------------------------
class DANN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, lambda_grl=1.0, num_dom=5, backbone=None, num_hidden_layers=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone)
        self.label_classifier = LabelClassifier(input_dim, hidden_dim, num_classes, num_hidden_layers)
        self.domain_discriminator = DomainDiscriminator(input_dim, hidden_dim, num_dom, num_hidden_layers)
        self.grl = GradientReversal(lambda_grl)
        self.label_classifier.apply(init_weights_xavier)
        self.domain_discriminator.apply(init_weights_xavier)


    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_preds = self.label_classifier(features)
        reversed_features = self.grl(features, alpha)
        domain_preds = self.domain_discriminator(reversed_features)
        return class_preds, domain_preds
