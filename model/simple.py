from torch import nn

class SoftmaxRegression(nn.Sequential):
    def __init__(self, num_inputs, num_classes, model_name):
        super().__init__(
            nn.Flatten(),
            nn.Linear(num_inputs, num_classes)
        )