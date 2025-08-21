import torch
from torch import nn
from torch.nn import functional as F
import dataclasses

# Create Model Configuration
dataclasses.dataclass
class ModelConfig:
    n_encoder_block: int = 6
    num_heads: int = 8
    embd_dim: int = 256
    drop_rate: int = 0.1

# create patch embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, num_patches: int, embd_dim: int):
        """
        img_size: size of the image (if the image size is (32, 32), the input will be 32)
        num_patches: How many patches you want to pass to your images
        embd_dim: number of embedding dimensions
        """
        super().__init__()
        
        self.num_patches = num_patches
        self.embd_dim = embd_dim
        
        kernel_size = img_size / num_patches
        if img_size % num_patches != 0:
            raise AttributeError(f"img_size % num_patches has to be 0, got {kernel_size}")
        kernel_size = img_size / num_patches
        self.proj = nn.Conv2d(in_channels = 3, 
                             out_channels = embd_dim, 
                             kernel_size = int(kernel_size),
                             stride = int(kernel_size))

        self.cls = nn.Embedding(1, embd_dim) # the cls token
        self.ppe = nn.Embedding(1 + num_patches ** 2, embd_dim) #the patch positional embeddings

    def forward(self, x):
        B, C, H, W = x.shape

        # get the patch embeddings
        x = self.proj(x) # (B, C, P, P)
        x = x.flatten(2).transpose(1, 2).contiguous() # (B, P^2, C)
        # concatenate with the cls token
        cls = self.cls(torch.full((B, 1), 0, dtype=torch.long).to(x.device))
        x = torch.cat((cls, x), dim = 1) # (B, 1 + P^2, C)
        # add positional embeddings
        x = x + self.ppe(torch.arange(start = 0, end = 1 + self.num_patches ** 2, device = x.device))
        return x

# Create multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embd_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.embd_dim = embd_dim
        self.c_attn = nn.Linear(embd_dim, 3 * embd_dim) # (B, T, C)
        self.c_proj = nn.Linear(embd_dim, embd_dim) # (B, T, C)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        # Ensure input is contiguous
        x = x.contiguous()

        # compute the query, key, value
        x = self.c_attn(x) # (B, T, C)
        q, k, v = x.split(self.embd_dim, dim = -1)

        # reshape the tensor to perform the multiheadattention
        q = q.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2) # (B, nh, T, c/nh)
        k = k.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2) # (B, nh, T, c/nh)
        v = v.view(B, T, self.num_heads, C//self.num_heads).transpose(1,2) # (B, nh, T, c/nh)

        # multiply q and k.T
        qk = torch.matmul(q, k.transpose(-1, -2)) / ((C//self.num_heads) ** 0.5) # (B, nh, T, T)

        qk = F.softmax(qk, dim= -1)
        output = torch.matmul(qk, v) # (B, nh, T, c/nh)
        output = output.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        output = self.c_proj(output)
        return output
    
# Create a encoder block
class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, embd_dim: int, drop_rate: int = 0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embd_dim)
        self.attn = MultiHeadAttention(num_heads, embd_dim)
        self.mlp = nn.Sequential(*[
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(4 * embd_dim, embd_dim)
        ])
        self.ln_2 = nn.LayerNorm(embd_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ln_2(self.mlp(x))
        return x
    
# Create Encoder Transformer Model
class VisualTransformer(nn.Module):
    def __init__(self, img_size, num_patches, config):
        """This is an Transformers Encoder Model. 
        """
        super().__init__()
        # fetch the parameters from the config
        n_encoder_block = config.n_encoder_block
        num_heads = config.num_heads
        embd_dim = config.embd_dim
        drop_rate = config.drop_rate

        self.pe = PatchEmbedding(img_size, num_patches, config.embd_dim)        
        self.h = nn.ModuleList(
            EncoderBlock(num_heads, embd_dim, drop_rate) for _ in range(n_encoder_block)
        )
        self.ln = nn.Linear(embd_dim, embd_dim)

    def forward(self, x: torch.Tensor):
        x = self.pe(x) # (B, 1 + P^2, embd_dim)
        for block in self.h:
            x = block(x) # (B, 1 + P^2, embd_dim)
        x = self.ln(x) # (B, 1 + P^2, embd_dim)
        return x[:, 0, :]
    
# Create Classification model based on the encoder block
class VisualTransformerClassifier(nn.Module):
    def __init__(self, img_size, num_patches, config):
        
        super().__init__()
        embd_dim = config.embd_dim
        drop_rate = config.drop_rate
        self.transformer = VisualTransformer(img_size, num_patches, config)
        self.classifier = nn.Sequential(*[
            nn.Linear(embd_dim, 2 * embd_dim),
            nn.Tanh(),
            nn.Linear(2* embd_dim, 10)
        ])

    def forward(self, x: torch.Tensor):
        """The input size will be (B, C, H, W). B is the number of batch and T is the block size"""

        #get the x shape
        B, C, H, W = x.shape

        # pass the model to the transformer
        x = self.transformer(x)
        x = self.classifier(x)
        return x