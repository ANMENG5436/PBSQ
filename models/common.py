"""Shared imports and constants for the split ViT/PBSQ implementation."""

import logging
import math
import os
import pdb
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from IPython import embed
from timm.models.layers import DropPath, trunc_normal_
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import linear, normalize

from face_pre_pro.mobilenet import MobileNetV3_backbone

MIN_NUM_PATCHES = 15
MAXIMUM_SINKHORN_ITERATIONS = 1000
