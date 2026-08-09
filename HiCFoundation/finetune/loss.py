import torch.nn as nn
#Please check https://pytorch.org/docs/stable/nn.html for more loss functions

class cosine_distance(nn.Module):
    def __init__(self):
        super(cosine_distance, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-08)
    def forward(self, x, y):
        return 1-self.cos(x,y)

def configure_loss(args):
    if args.loss_type == 1:
        return nn.MSELoss()
    elif args.loss_type == 2:
        return cosine_distance()
    else:
        raise Exception("Unknown loss type: {}".format(args.loss_type)) 



