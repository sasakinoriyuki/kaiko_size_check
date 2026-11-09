# Extracted from YOLOv8-DSRI, Ultralytics AGPL-3.0 licensed source.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SegmentationModel(nn.Module):
    def forward(self, x):
        saved=[]
        for module in self.model:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f,int) else [x if j == -1 else saved[j] for j in module.f]
            x=module(x)
            saved.append(x if module.i in self.save else None)
        return x
