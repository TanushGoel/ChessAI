import torch
from torchviz import make_dot

import warnings
warnings.filterwarnings("ignore")

class Model(torch.nn.Module):

    def __init__(self, game, device, num_layers=3):
        super(Model, self).__init__()
        
        self.game = game
        self.device = device
        
        self.channels = game().get_state().size(0)
        self.image_size = game().get_state().size(1)
        action_space = game.get_action_space()

        self.resnet = torch.nn.ModuleList(ResBlock(self.channels) for _ in range(num_layers))
        self.patchize = ViTPatch(channels=self.channels, image_size=self.image_size, patch_size=1, embed_dim=self.channels)
        self.transformer = torch.nn.ModuleList(TransformerBlock(embed_dim=self.channels, num_heads=self.channels, mlp_dim=256, dropout=0.1) for _ in range(num_layers))

        self.action = torch.nn.Sequential(
            torch.nn.Linear(256, 1024), torch.nn.GELU(),
            torch.nn.Linear(1024, 512), torch.nn.GELU(),
            torch.nn.Linear(512, action_space)
        )

        self.value = torch.nn.Sequential(
            torch.nn.Linear(256, 256), torch.nn.GELU(),
            torch.nn.Linear(256, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 1), torch.nn.Tanh()
        )
        
        self.to(self.device)
        self.load()
        
    def forward(self, x):
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        x = self.patchize(x)
        for res, vit in zip(self.resnet, self.transformer):
            y = vit(x)
            x = x.reshape(batch_size, self.channels, self.image_size, self.image_size)
            z = res(x)
            z = z.reshape(batch_size, -1, self.channels)
            x = y + z
        x = x.view(batch_size, -1)
        return self.action(x), self.value(x)

    # def forward(self, x):
    #     x = x.to(self.device)
    #     if x.dim() == 3:
    #         x = x.unsqueeze(0)
    #     batch_size = x.size(0)
    #     y = self.patchize(x)
    #     for res in self.resnet:
    #         x = res(x)
    #     for vit in self.transformer:
    #         y = vit(y)
    #     x = x.view(batch_size, self.channels, -1)
    #     z = x + y     # z = torch.cat((x, y), dim=2)
    #     z = z.view(batch_size, -1)
    #     return self.action(z), self.value(z)
    
    def evaluate(self, game):
        policy, value = self(game.get_state())
        value = value.item()
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().detach().numpy()
        policy *= game.get_legal_moves()
        policy /= policy.sum()
        return policy, value
    
    def save(self, path="model"):
        torch.save(self.state_dict(), f"data/{path}.pt")

    def load(self, path="model"):
        try:
            self.load_state_dict(torch.load(f"data/{path}.pt"))
        except:
            print("No model found, initializing new model...")
            self.initialize()
        self.eval()
        
    def initialize(self):
        for param in self.parameters():
            if param.dim() > 1: # weight
                torch.nn.init.xavier_uniform_(param)
            else: # bias
                torch.nn.init.uniform_(param)
        self.count_params()
    
    def count_params(self):
        print(f"{sum(p.numel() for p in self.parameters()):,} parameters")
                
    def show(self):
        return make_dot(self(self.game().get_state().unsqueeze(0).to(self.device)), params=dict(list(self.named_parameters())))


class ResBlock(torch.nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        x = torch.nn.functional.gelu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        x = torch.nn.functional.gelu(x)
        return x
    

class MultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = torch.nn.Linear(embed_dim, embed_dim)
        self.key_dense = torch.nn.Linear(embed_dim, embed_dim)
        self.value_dense = torch.nn.Linear(embed_dim, embed_dim)
        self.combine_heads = torch.nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        weights = torch.nn.functional.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.transpose(1, 2)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = attention.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.combine_heads(attention)
        return output


class TransformerBlock(torch.nn.Module):

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, mlp_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_dim, embed_dim),
            torch.nn.Dropout(dropout)
        )
        self.layernorm1 = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, inputs):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + inputs
        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output)
        return mlp_output + out1


class ViTPatch(torch.nn.Module):

    def __init__(self, channels, image_size, patch_size, embed_dim):
        super(ViTPatch, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size

        self.rescale = torch.nn.Sequential(
            torch.nn.LayerNorm((channels, image_size, image_size)),
            torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1),
        )
        self.pos_emb = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.class_emb = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_proj = torch.nn.Linear(self.patch_dim, embed_dim)


    def extract_patches(self, images):
        batch_size = images.size(0)
        patches = torch.nn.functional.unfold(images, self.patch_size, stride=self.patch_size, padding=0)
        patches = patches.transpose(1, 2).reshape(batch_size, -1, self.patch_dim)
        return patches

    def forward(self, x):
        x = self.rescale(x)
        patches = self.extract_patches(x)
        x = self.patch_proj(patches)
        x = x + self.pos_emb
        return x