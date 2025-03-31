import torch
import torch.nn as nn
from torch.autograd import Function
from utils import init_weights_xavier
from torchvision.models import resnet18, resnet34

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
    def __init__(self):
        super().__init__()
        self.net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # Freeze the feature extractor
        for param in self.net.parameters():
            param.requires_grad = False
        # add three linear layers to the end of the network
        self.net.head = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Linear(512, 512))
        
    def forward(self, x):
        return self.net(x)

# --------------------------
# Label Classifier
# --------------------------
class LabelClassifier(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):
        return self.classifier(features)

# --------------------------
# Domain Discriminator
# --------------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim = 512, hidden_dim=256, num_dom=5):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, num_dom)
        )

    def forward(self, features):
        return self.discriminator(features)

# --------------------------
# DANN Model Wrapper
# --------------------------
class DANN(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, lambda_grl=1.0, num_dom=5):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.label_classifier = LabelClassifier(input_dim, hidden_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(input_dim, hidden_dim, num_dom)
        self.grl = GradientReversal(lambda_grl)
        self.label_classifier.apply(init_weights_xavier)
        self.domain_discriminator.apply(init_weights_xavier)


    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_preds = self.label_classifier(features)
        reversed_features = self.grl(features, alpha)
        domain_preds = self.domain_discriminator(reversed_features)
        return class_preds, domain_preds
