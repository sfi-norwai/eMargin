from src.models.attention_model import *

import torch
import torch.nn as nn
from pathlib import Path
from einops import rearrange

from src.layers.Embed import DataEmbedding, Patching
from src.layers.RevIN import RevIN
from src.layers.einops_modules import RearrangeModule
    


class TimeDRL_Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.C = input_size
        self.patch_len = 10
        self.stride = 1
        self.enable_channel_independence = False
        self.d_model = input_size
        self.T_p = 300
        self.dropout = 0.2
        self.get_i = "cls"
        self.i_dim = input_size



        # RevIN (without affine transformation)
        self.revin = RevIN(self.C, affine=False)

        # Input layer
        self._set_input_layer()

        # Pretext layer (for predictive and contrastive tasks)
        self._set_pretext_layer()


        # Encoder
        self.encoder = FeatureProjector(input_size=input_size, output_size=output_size)

        # [CLS] token (we need this no matter what `get_i` is)
        if self.enable_channel_independence:
            self.cls_token = nn.Parameter(
                torch.randn(1, self.C, self.patch_len)
            )
        else:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, self.C * self.patch_len)
            )

    def _set_input_layer(self):
        self.patching = Patching(
            self.patch_len, self.stride, self.enable_channel_independence
        )  # (B, T_in, C) -> (B * C, T_p, P) (Enable CI) or (B, T_p, C * P) (Disable CI)
        if self.enable_channel_independence:
            self.input_layer = DataEmbedding(
                last_dim=self.patch_len,
                d_model=self.d_model,
                dropout=self.dropout,
                pos_embed_type= "fixed",
                token_embed_type= "linear",
                token_embed_kernel_size= 3,
            )  # (B * C, T_p, P) -> (B * C, T_p, D)
        else:
            self.input_layer = DataEmbedding(
                last_dim=self.C * self.patch_len,
                d_model=self.d_model,
                dropout=self.dropout,
                pos_embed_type= "fixed",
                token_embed_type= "linear",
                token_embed_kernel_size= 3,
            )  # (B, T_p, C * P) -> (B, T_p, D)

    def _set_pretext_layer(self):
        # Predictive task
        if self.enable_channel_independence:
            self.predictive_linear = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.output_size, self.patch_len),
            )  # (B * C, T_p, D) -> (B * C, T_p, P)
        else:
            self.predictive_linear = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.output_size, self.C * self.patch_len),
            )  # (B, T_p, D) -> (B, T_p, C * P)

        # (For contrastive task) set i_dim based on get_i,
        # and set additional layers if necessary
        if self.get_i == "cls":
            assert self.i_dim == self.d_model
        elif self.get_i == "last":
            assert self.i_dim == self.d_model
        elif self.get_i == "gap":
            assert self.i_dim == self.d_model
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif self.get_i == "all":
            assert self.i_dim == self.T_p * self.d_model
            if self.enable_channel_independence:
                self.flatten = RearrangeModule(
                    "(B C) T_p D -> (B C) (T_p D)",
                    C=self.C,
                    T_p=self.T_p,
                    D=self.d_model,
                )
            else:
                self.flatten = RearrangeModule(
                    "B T_p D -> B (T_p D)",
                    T_p=self.T_p,
                    D=self.d_model,
                )

        # Contrastive task
        self.contrastive_predictor = nn.Sequential(
            nn.Linear(self.output_size, self.i_dim // 2),
            nn.BatchNorm1d(self.i_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.i_dim // 2, self.output_size),
        )  # (B * C, i_dim) -> (B * C, i_dim)

    def forward(self, x):  # (B, T_in, C)
        B, T_in, C = x.shape

        x = x.float()

        # Instance Normalization
        x = self.revin(x, "norm")  # (B, T_in, C)
        

        # Create two data (it should be the same if `data_aug` is `none`)
       
        # dropout randomness
        x_1 = x
        x_2 = x

        # Patching
        x_1 = self.patching(
            x_1
        )  # (B * C, T_p, P) (Enable CI) or (B, T_p, C * P) (Disable CI)
        x_2 = self.patching(x_2)

        # [CLS] token
        
        if self.enable_channel_independence:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, C, P)
            cls_token = rearrange(cls_token, "B C P -> (B C) 1 P")  # (B * C, 1, P)
            x_1 = torch.cat([cls_token, x_1], dim=1)  # (B * C, T_p + 1, P)
            x_2 = torch.cat([cls_token, x_2], dim=1)
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, C * P)
            x_1 = torch.cat([cls_token, x_1], dim=1)  # (B, T_p + 1, C * P)
            x_2 = torch.cat([cls_token, x_2], dim=1)

        # First pass
        x_1 = self.input_layer(x_1)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)

        z_1 = self.encoder(x_1)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)

        # Second pass
        x_2 = self.input_layer(x_2)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)
        z_2 = self.encoder(x_2)  # (B * C, T_p + 1, D) or (B, T_p + 1, D)

        # Predictive task
        t_1 = z_1[:, 1:, :]  # (B * C, T_p, D) or (B, T_p, D)
        t_2 = z_2[:, 1:, :]  # (B * C, T_p, D) or (B, T_p, D)
        x_pred_1 = self.predictive_linear(t_1)  # (B * C, T_p, P) or (B, T_p, C * P)
        x_pred_2 = self.predictive_linear(t_2)  # (B * C, T_p, P) or (B, T_p, C * P)

        # Contrastive task
        if self.get_i == "cls":
            i_1 = z_1[:, 0, :]  # (B * C, D) or (B, D)
            i_2 = z_2[:, 0, :]  # (B * C, D) or (B, D)
        elif self.get_i == "last":
            i_1 = t_1[:, -1, :]  # (B * C, D) or (B, D)
            i_2 = t_2[:, -1, :]
        elif self.get_i == "gap":
            i_1 = self.gap(t_1.transpose(1, 2)).squeeze(-1)  # (B * C, D) or (B, D)
            i_2 = self.gap(t_2.transpose(1, 2)).squeeze(-1)  # (B * C, D) or (B, D)
        elif self.get_i == "all":
            i_1 = self.flatten(t_1)  # (B * C, T_p * D) or (B, T_p * D)
            i_2 = self.flatten(t_2)  # (B * C, T_p * D) or (B, T_p * D)
        else:
            raise NotImplementedError
        i_1_pred = self.contrastive_predictor(i_1)  # (B * C, i_dim) or (B, i_dim)
        i_2_pred = self.contrastive_predictor(i_2)  # (B * C, i_dim) or (B, i_dim)

        
        return (
            t_1,  # (B * C, T_p, D) or (B, T_p, D)
            t_2,  # (B * C, T_p, D) or (B, T_p, D)
            x_pred_1,  # (B * C, T_p, P) or (B, T_p, C * P)
            x_pred_2,  # (B * C, T_p, P) or (B, T_p, C * P)
            i_1,  # (B * C, i_dim) or (B, i_dim)
            i_2,  # (B * C, i_dim) or (B, i_dim)
            i_1_pred,  # (B * C, i_dim) or (B, i_dim)
            i_2_pred,  # (B * C, i_dim) or (B, i_dim)
        )
