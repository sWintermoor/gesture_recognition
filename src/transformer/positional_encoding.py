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
            T = x_input.size(1)
            d_model = x_input.size(2)

            pos = torch.arange(T, device=x_input.device).unsqueeze(1)

            pos_encode = torch.zeros((T, d_model), device=x_input.device)   
            
            pos_encode[:, 0::2] = torch.sin(pos * self.omega[:d_model//2])
            pos_encode[:, 1::2] = torch.cos(pos * self.omega[:d_model - d_model//2])

            pos_encode = x_input + pos_encode.unsqueeze(0)

        else:
            raise ValueError(f"Wrong input dimension. Your dimension is: {x_input.dim()}, but should be: 3")
        
        return pos_encode