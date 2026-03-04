import torch
import torch.nn as nn

# Sinusoidal
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        pos_encode_range = d_model // 2
        omega = torch.zeros(pos_encode_range,)
        for i in range(pos_encode_range):
            omega[i] = 10000**(-2 * i / d_model)

        self.register_buffer("omega", omega)


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        pos_encode = torch.zeros_like(x_input)
        if(x_input.dim() == 3):
            for pos in range(x_input.size(1)):
                for i in range(x_input.size(2)):
                    if i % 2 == 0:
                        pos_encode[:, pos, i] = x_input[:, pos, i] + torch.sin(pos * self.omega[i//2])
                    else:
                        pos_encode[:, pos, i] = x_input[:, pos, i] + torch.cos(pos * self.omega[(i-1)//2])
        else:
            raise ValueError(f"Wrong input dimension. Your dimension is: {x_input.dim()}, but should be: 3")
        
        return pos_encode