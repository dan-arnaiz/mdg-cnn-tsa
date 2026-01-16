import torch
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/cnn_tsa/corr1_k32/lr1e-05/main/best_weights.pt"
)
CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models/cnn_tsa/corr1_k32/lr1e-05/main/config.json"
)

print("=" * 70)
print("DIAGNOSTIC REPORT FOR CNN-TSA MODEL")
print("=" * 70)

# 1. Load and display config.json
print("\n[1] CONFIG.JSON CONTENTS:")
print("-" * 70)
with open(CONFIG_PATH) as f:
    cfg = json.load(f)
    for key, value in cfg.items():
        print(f"  {key}: {value}")

# 2. Load checkpoint
print("\n[2] LOADING CHECKPOINT:")
print("-" * 70)
try:
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    print(f"  ✓ Checkpoint loaded successfully")
    print(f"  Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"  Dictionary keys: {list(checkpoint.keys())}")
except Exception as e:
    print(f"  ✗ Error loading checkpoint: {e}")
    exit(1)

# 3. Extract state_dict
print("\n[3] EXTRACTING STATE_DICT:")
print("-" * 70)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("  ✓ Using checkpoint['model_state_dict']")
else:
    state_dict = checkpoint
    print("  ✓ Using checkpoint directly as state_dict")

# 4. Display all layer shapes
print("\n[4] MODEL LAYER SHAPES:")
print("-" * 70)
for key, value in state_dict.items():
    print(f"  {key:40s} → {str(value.shape):30s}")

# 5. Analyze critical layers
print("\n[5] CRITICAL LAYER ANALYSIS:")
print("-" * 70)

# Conv layers
if 'conv1.weight' in state_dict:
    conv1_shape = state_dict['conv1.weight'].shape
    print(f"  conv1.weight: {conv1_shape}")
    print(f"    Expected format: (out_channels, in_channels, kernel_size)")
    print(f"    → out_channels={conv1_shape[0]}, in_channels={conv1_shape[1]}, kernel={conv1_shape[2]}")
    
if 'conv2.weight' in state_dict:
    conv2_shape = state_dict['conv2.weight'].shape
    print(f"\n  conv2.weight: {conv2_shape}")
    print(f"    Expected format: (out_channels, in_channels, kernel_size)")
    print(f"    → out_channels={conv2_shape[0]}, in_channels={conv2_shape[1]}, kernel={conv2_shape[2]}")

# Attention layer
if 'mhsa.in_proj_weight' in state_dict:
    mhsa_shape = state_dict['mhsa.in_proj_weight'].shape
    print(f"\n  mhsa.in_proj_weight: {mhsa_shape}")
    embed_dim = mhsa_shape[1]
    total_dim = mhsa_shape[0]
    num_heads_calc = embed_dim // 32  # Typical calculation
    print(f"    embed_dim={embed_dim}, total_proj_dim={total_dim}")
    print(f"    This suggests num_heads could be: 2, 4, or 8 (depending on implementation)")

# FFN layers
print(f"\n  FFN Structure:")
ffn_layers = [k for k in state_dict.keys() if k.startswith('ffn.')]
for layer in sorted(ffn_layers):
    print(f"    {layer}: {state_dict[layer].shape}")

# 6. Generate correct architecture
print("\n[6] RECOMMENDED MODEL ARCHITECTURE:")
print("-" * 70)

if 'conv1.weight' in state_dict and 'conv2.weight' in state_dict:
    conv1_shape = state_dict['conv1.weight'].shape
    conv2_shape = state_dict['conv2.weight'].shape
    
    print("class CNNTSA(nn.Module):")
    print("    def __init__(self):")
    print("        super().__init__()")
    print()
    print("        # CNN feature extractor")
    print(f"        self.conv1 = nn.Conv1d({conv1_shape[1]}, {conv1_shape[0]}, kernel_size={conv1_shape[2]}, padding={(conv1_shape[2]-1)//2})")
    print(f"        self.conv2 = nn.Conv1d({conv2_shape[1]}, {conv2_shape[0]}, kernel_size={conv2_shape[2]}, padding={(conv2_shape[2]-1)//2})")
    print("        self.relu = nn.ReLU()")
    print()
    
    if 'mhsa.in_proj_weight' in state_dict:
        embed_dim = state_dict['mhsa.in_proj_weight'].shape[1]
        print("        # Transformer-style TSA block")
        print(f"        self.mhsa = nn.MultiheadAttention(")
        print(f"            embed_dim={embed_dim},")
        print(f"            num_heads=2,  # or 4 - try both")
        print(f"            batch_first=True")
        print(f"        )")
        print(f"        self.norm1 = nn.LayerNorm({embed_dim})")
    
    print()
    print("        # FFN block")
    ffn_layers = sorted([k for k in state_dict.keys() if k.startswith('ffn.') and 'weight' in k])
    if ffn_layers:
        print("        self.ffn = nn.Sequential(")
        for i, layer_name in enumerate(ffn_layers):
            layer_shape = state_dict[layer_name].shape
            if len(layer_shape) == 2:  # Linear layer
                print(f"            nn.Linear({layer_shape[1]}, {layer_shape[0]}),  # {layer_name}")
                # Check if next layer exists to add activation
                next_idx = int(layer_name.split('.')[1]) + 1
                if f'ffn.{next_idx}.weight' in state_dict or f'ffn.{next_idx}.bias' in state_dict:
                    print(f"            nn.ReLU(),")
                    if next_idx + 1 < len(ffn_layers):
                        print(f"            nn.Dropout(0.1),")
        print("        )")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("1. Review the 'RECOMMENDED MODEL ARCHITECTURE' section above")
print("2. Copy the recommended architecture to your controller.py")
print("3. Pay special attention to conv1 and conv2 parameters")
print("4. Try num_heads=2 first, if that fails try num_heads=4")