

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .resnet18 import ResNet18
import torch 
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.utils.model_zoo as model_zoo
#from torch.nn import LayerNorm
from copy import deepcopy
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model
from timm.models.vision_transformer import Mlp, DropPath
import numpy as np
_logger = logging.getLogger(__name__)

__all__ = [
    'HTR_VT_16_heads_with_headSE', 
    'HTR_VT_24_heads_with_headSE',
]




class HeadSEGate(nn.Module):
    def __init__(self, num_heads, reduction=4):
        super().__init__()
        hidden = max(1, num_heads // reduction)
        self.fc1 = nn.Linear(num_heads, hidden)
        self.fc2 = nn.Linear(hidden, num_heads)

    def forward(self, u):
        # u: [B, heads, N, d]
        B, H, N, D = u.shape

        # Global average pooling over tokens and head_dim
        z = u.mean(dim=(2, 3))  # [B, H]

        w = F.relu(self.fc1(z))
        w = torch.sigmoid(self.fc2(w))  # [B, H]

        # Rescale heads
        u = u * w.unsqueeze(-1).unsqueeze(-1)

        return u


class LayerNorm(nn.Module):

    def forward(self, x):
            # print(f"LayerNorm forward Input device: {x.device}")  # Debug
            # print(f"LayerNorm forward Model device: {next(self.parameters()).device}")
            return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

def create_vitstr(num_tokens, model=None, checkpoint_path='', **kwargs):
    # Special handling for the MAE custom model
    if model == 'vitstr_mae_custom':
        vitstr = create_model(
            model,
            pretrained=False,  # No pretrained weights available by default
            nb_cls=num_tokens,
            checkpoint_path=checkpoint_path,
            **kwargs)
    else:
        vitstr = create_model(
            model,
            pretrained=True,
            num_classes=num_tokens,
            checkpoint_path=checkpoint_path,
            **kwargs)

    # Reset classifier to ensure proper initialization
    vitstr.reset_classifier(num_classes=num_tokens)

    return vitstr

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_patches,
                 mlp_ratio=4., qkv_bias=False,
                 drop=0.0, attn_drop=0.,
                 init_values=None, drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_head_se=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = Attention(dim, num_patches,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              use_head_se=use_head_se)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                      act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # print(f"Block forward Input device: {x.device}")  # Debug
        # print(f"Block forward Model device: {next(self.parameters()).device}")
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, 
                 qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_head_se=False, reduction=4):
        super().__init__()
        self.use_head_se = use_head_se
        if self.use_head_se:
            self.head_se = HeadSEGate(num_heads, reduction=reduction)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # print(f"Class Attention forward Input device: {x.device}")  # Debug
        # print(f"Class Attention forward Model device: {next(self.parameters()).device}")

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        u = attn @ v   # [B, heads, N, d]

        if self.use_head_se:
            u = self.head_se(u)

        x = u.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # print(f"LayerScale forward Input device: {x.device}")  # Debug
        # print(f"LayerScale forward Model device: {next(self.parameters()).device}")
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: (int, int) of the grid height and width
    return:
    pos_embed: [grid_size[0]*grid_size[1], embed_dim]
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb






# class MaskedAutoencoderViT(nn.Module):
#     def __init__(self, nb_cls, img_size=(224, 224), patch_size=(16, 16), embed_dim=768,
#                  depth=4, num_heads=6, mlp_ratio=4., norm_layer=nn.LayerNorm,
#                  mask_ratio=0.75, max_span_length=3, use_masking=False):
#         super().__init__()
        
#         # Store device info
#         self._device = torch.device('cpu')
#         #self.grid_size = (4, 56)  # Hardcode based on ResNet18 output
#         self.num_patches = self.grid_size[0] * self.grid_size[1]  # 224 patches

#         # Initialize positional embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
#         # Store essential dimensions first
#         self.embed_dim = embed_dim
#         self.num_classes = nb_cls
#         self.mask_ratio = mask_ratio
#         self.max_span_length = max_span_length
#         self.use_masking = use_masking
        
#         # Initialize layers
#         self.layer_norm = nn.LayerNorm(embed_dim)
#         self.patch_embed = ResNet18(embed_dim)
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         print(f"Grid size: {self.grid_size}")
        
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         print(f"Num patches: {self.num_patches}")
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
#         print(f"Positional embeddings shape: {self.pos_embed.shape}")
#         # Should output [batch, 768, 4, 56] for 224x224 input
#         print(f"Confirm ResNet18's output shape:{self.patch_embed(torch.randn(1, 1, 224, 224)).shape}")
#         # Transformer blocks
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, self.num_patches, mlp_ratio, 
#                  qkv_bias=True, norm_layer=norm_layer)
#             for _ in range(depth)])
        
#         self.norm = norm_layer(embed_dim)
#         self.head = nn.Linear(embed_dim, nb_cls)
        
#         # Initialize weights
#         self.initialize_weights()

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, nb_cls, img_size=(224, 224), patch_size=(16, 16), embed_dim=768,
                 depth=4, num_heads=6, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 mask_ratio=0.75, max_span_length=3, use_masking=False,use_head_se=False):
        super().__init__()
        
        # Original configuration preserved
        self._device = torch.device('cpu')
        self.use_head_se = use_head_se
        self.grid_size = (14, 14)
        self.num_patches = 14 * 14
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.embed_dim = embed_dim
        self.num_classes = nb_cls
        self.mask_ratio = mask_ratio
        self.max_span_length = max_span_length
        self.use_masking = use_masking
        self.adaptive_pool = nn.AdaptiveAvgPool1d(25) 
        
        # Original layers preserved
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.patch_embed = ResNet18(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # +1 for cls_token
        
        # Transformer blocks
        '''
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches + 1, mlp_ratio,  # +1 for cls_token
                 qkv_bias=True, norm_layer=norm_layer)
         '''
        self.blocks = nn.ModuleList([Block(embed_dim,num_heads,self.num_patches + 1,mlp_ratio,qkv_bias=True,norm_layer=norm_layer,use_head_se=use_head_se)
        for _ in range(depth)])

        
        self.norm = norm_layer(embed_dim)
        #self.head = nn.Linear(embed_dim, nb_cls)

        
        
        self.initialize_weights()

    def reset_classifier(self, num_classes):
        
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        #print(f"inside reset classifier: {self.embed_dim, num_classes}")
        #self.head = nn.Linear(768,96)
        #self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
  
    def to(self, device):
        self._device = device
        return super().to(device)
        

    def initialize_weights(self):
        nn.init.normal_(self.cls_token, std=.02)
       
       
        # Generate positional embeddings for the correct grid size
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=self.grid_size  # (4, 56)
        )
        # Add positional embedding for cls_token (zeros)
        cls_pos_embed = np.zeros((1, self.embed_dim))
        pos_embed = np.concatenate([cls_pos_embed, pos_embed], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


    
        nn.init.normal_(self.mask_token, std=.02)
        
        # Apply custom initialization
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:,idx:idx + max_span_length,:] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked



    
    def forward(self, x, seqlen: int =25, mask_ratio=None, max_span_length=None, use_masking=None):
        # print(f"MaskedAutoencoderViT forward Input device: {x.device}")  # Debug
        # print(f"MaskedAutoencoderViT forward Model device: {next(self.parameters()).device}")
        if not any(p.is_cuda for p in self.parameters()):
            # Model is on CPU
            if x.is_cuda:
                x = x.cpu()
        else:
            # Model is on GPU
            if not x.is_cuda:
                x = x.cuda()

        # if next(self.parameters()).is_cuda and not x.is_cuda:
        #     x = x.cuda()
        # elif not next(self.parameters()).is_cuda and x.is_cuda:
        #     x = x.cpu()
        # Original patch embedding
        #print(f"forwarrrrrrrrrd function parameters:{seqlen}")
        #print(f"MaskedAutoEncoderViT forward method Input shape:{x.shape}")
        x = self.patch_embed(x)  # [b, 768, 14, 14]
        #print(f"After patch_embed():{x.shape}")
        x = x.flatten(2).transpose(1, 2)  # [b, 196, 768]
        #print(f"After flatten:{x.shape}")
        x = self.layer_norm(x)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [b, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # [b, 197, 768]
        #print(f"After adding cls_token: {x.shape}")
        # masking: length -> length * mask_ratio
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)        
        x = x + self.pos_embed
        #print(f"After pos_embed:{x.shape}")
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        #print(f"After encoder blocks:{x.shape}")
        x = self.norm(x)
        #print(f"before selectig 25: {x.shape}")
        
        # FIX: Select only the first 'seqlen' tokens for text recognition
        # This maintains patch order while fixing sequence length
        #print(f"sequence length: {seqlen}")
        x = x[:, :seqlen]  # [b, 25, 768]
        #print(f"after selectig 25: {x.shape}")
        # Add validation
        # if x.shape[1] != seqlen:
        #     raise ValueError(f"Sequence length mismatch: {x.shape[1]} != {seqlen}")     
        b, s, e = x.size()

        x = x.reshape(b*s, e)
        #print(f"after reshape: {x.shape}")
        x = self.head(x).view(b, s, self.num_classes)
        return x
        
    

       
   
class ViTSTR(VisionTransformer):
    '''
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = kwargs.get('mask_ratio', 0.0)#added
        self.use_masking = kwargs.get('use_masking', False)  #added 
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # print(f"ViTSTR forward_features Input device: {x.device}")  # Debug
        # print(f"ViTSTR forward_features Model device: {next(self.parameters()).device}")
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio#added
        use_masking = self.use_masking if use_masking is None else use_masking  #added      
        B = x.shape[0]
        
        x = self.patch_embed(x)
        

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, seqlen: int =25):
        # print(f"ViTstr forward Input device: {x.device}")  # Debug
        # print(f"ViTstr forward Model device: {next(self.parameters()).device}")
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b*s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    '''
    Loads a pretrained checkpoint
    From an older version of timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

   
    
@register_model
def HTR_VT_16_heads_with_headSE(pretrained=False, **kwargs):
    default_kwargs = {
        'img_size': (224, 224),
        'patch_size': (4, 64),
        'embed_dim': 768,
        'depth': 4,
        'num_heads': 16,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'in_chans': 1,
        'use_head_se': True
    }

    default_kwargs.update(kwargs)

    model = MaskedAutoencoderViT(
        nb_cls=default_kwargs.get('num_classes', 96),
        img_size=default_kwargs['img_size'],
        patch_size=default_kwargs['patch_size'],
        embed_dim=default_kwargs['embed_dim'],
        depth=default_kwargs['depth'],
        num_heads=default_kwargs['num_heads'],
        mlp_ratio=default_kwargs['mlp_ratio'],
        norm_layer=default_kwargs['norm_layer'],
        use_head_se=default_kwargs['use_head_se']
    )

    return model
    
@register_model
def HTR_VT_24_heads_with_headSE(pretrained=False, **kwargs):
    default_kwargs = {
        'img_size': (224, 224),
        'patch_size': (4, 64),
        'embed_dim': 768,
        'depth': 4,
        'num_heads': 24,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'in_chans': 1,
        'use_head_se': True
    }

    default_kwargs.update(kwargs)

    model = MaskedAutoencoderViT(
        nb_cls=default_kwargs.get('num_classes', 96),
        img_size=default_kwargs['img_size'],
        patch_size=default_kwargs['patch_size'],
        embed_dim=default_kwargs['embed_dim'],
        depth=default_kwargs['depth'],
        num_heads=default_kwargs['num_heads'],
        mlp_ratio=default_kwargs['mlp_ratio'],
        norm_layer=default_kwargs['norm_layer'],
        use_head_se=default_kwargs['use_head_se']
    )

    return model
