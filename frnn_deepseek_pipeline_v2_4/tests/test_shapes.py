import torch;from frnn_path_b import FRNNPathB; m=FRNNPathB(128,64,512,256,512,128); x=torch.randn(2,8,128); y,modes=m(x); assert y.shape==(2,8,64)
