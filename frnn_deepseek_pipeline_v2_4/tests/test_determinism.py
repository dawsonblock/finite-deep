import torch;from frnn_path_b import FRNNPathB; m=FRNNPathB(128,64,512,256,512,128);m.eval();x=torch.randn(2,8,128);y1,_=m(x);y2,_=m(x);assert torch.allclose(y1,y2)
