import torch
import torch.nn as nn
import lightning as L

from .positional_encoding import SinusoidalPositionalEncoding
from .encoding_layer import EncodingLayer

class KeypointTransformer(nn.Module):
    def __init__(self, 
                 input_size: int, # Input featuresize 
                 d_model: int, # Dimension of model -> Token size
                 num_classes: int,
                 nheads: int = 4, # Number of attention heads for transformer
                 dim_ff: int = 512, 
                 #batch_size: int = 32,
                 dropout_p: float = 0.1,
                 nlayers: int = 6
                 ):
        super().__init__()

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model)
        )

        self.pos_encode = SinusoidalPositionalEncoding(d_model=d_model)

        self.layers = nn.ModuleList(
            [EncodingLayer(nheads=nheads, d_model=d_model, dim_ff=dim_ff, dropout_p=dropout_p) for _ in range(nlayers)]
        )

        self.final_norm = nn.Linear(d_model, num_classes)
        

    def forward(self, x_input, mask=None, return_attentions=False):
        x = self.input_proj(x_input)

        batch_size = x.size(0)
        cls = self.cls.expand(batch_size, -1, -1)
        x = torch.concat([cls, x], dim=1)

        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        attention_weights = []
        x = self.pos_encode(x)
        for layer in self.layers:
            x, attn = layer(x, mask=mask)
            attention_weights.append(attn)

        cls = x[:, 0, :]
        logits = self.final_norm(cls)

        if return_attentions:
            return logits, attention_weights
        
        return logits


class LitKeypointTransformer(L.LightningModule):
    def __init__(self, input_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.save_hyperparameters()

        self.model = KeypointTransformer(input_size=input_size, d_model=d_model, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_input, mask=None):
        return self.model(x_input, mask=mask)
    
    def training_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask=~mask) # inverting because of torch logic
        loss = self.loss_fn(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask, y = batch
        y_hat = self(x, mask=~mask)
        loss = self.loss_fn(y_hat, y)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)   