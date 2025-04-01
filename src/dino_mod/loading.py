import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


from utils import CombinedH5UnlabeledDataset, ClassDiscriminator, GradientReversal, AlphaScheduler

student = vits.__dict__['vit_small'](
            patch_size=16,
            drop_path_rate=0.1,  # stochastic depth
        )

teacher = vits.__dict__['vit_small'](
            patch_size=16,
            drop_path_rate=0.1,  # stochastic depth
        )
to_restore = {"epoch": 0}
utils.restart_from_checkpoint(
        os.path.join('../../checkpoints/', "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
    )
print(student)