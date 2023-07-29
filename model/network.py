import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchinfo
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,64,5)
        self.conv2=nn.Conv2d(64,16,5)
        self.fc1=nn.Linear(16*53*53,512)
        self.fc2=nn.Linear(512,320)
        self.fc3=nn.Linear(320,256)
              
    def forward(self,t):
        t=F.max_pool2d(F.relu(self.conv1(t)),2)
        t=F.max_pool2d(F.relu(self.conv2(t)),2)
        t=torch.flatten(t,1)
        t=F.relu(self.fc1(t))
        t=F.relu(self.fc2(t))
        t=self.fc3(t)
        return t
    
class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=2, 
                                  end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        return x_flattened.permute(0, 2, 1) 

class ViT(nn.Module): 
  def __init__(self,
               img_size=224, 
               num_channels=3,
               patch_size=16,
               embedding_dim=768,
               dropout=0.1, 
               mlp_size=3072, 
               num_transformer_layers=12, 
               num_heads=12, 
               num_classes=257): 
    super().__init__()

    assert img_size % patch_size == 0, "Image size must be divisble by patch size."
    self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)
    self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                    requires_grad=True)

    num_patches = (img_size * img_size) // patch_size**2 # N = HW/P**2
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))
    self.embedding_dropout = nn.Dropout(p=dropout)
    self.norm = nn.LayerNorm(normalized_shape=embedding_dim) 
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                              nhead=num_heads,
                                                                                              dim_feedforward=mlp_size,
                                                                                              activation="gelu"), # Create a single Transformer Encoder Layer
                                                     num_layers=num_transformer_layers) # Stack it N times

    self.mlp_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self, x):
    batch_size = x.shape[0]
    x = self.patch_embedding(x)
    class_token = self.class_token.expand(batch_size, -1, -1) 
    x = torch.cat((class_token, x), dim=1)
    x = self.positional_embedding + x
    x = self.embedding_dropout(x)
    x = self.norm(x)  # normalize the features
    x = x.permute(1,0,2)
    x = self.transformer_encoder(x)
    x = self.mlp_head(x[:, 0])
    return x



