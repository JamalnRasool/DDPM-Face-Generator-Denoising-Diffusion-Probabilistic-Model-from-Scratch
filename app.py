import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from huggingface_hub import hf_hub_download

# ── Model Architecture (same as your notebook) ──────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_emb_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
    def forward(self, x, t):
        x = self.res(x, t)
        return self.down(x), x

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + skip_ch, out_ch, time_emb_dim)
    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        ch1, ch2, ch3 = base_ch, base_ch*2, base_ch*4
        self.init_conv = nn.Conv2d(in_ch, ch1, 3, padding=1)
        self.down1 = Down(ch1, ch2, time_emb_dim)
        self.down2 = Down(ch2, ch3, time_emb_dim)
        self.down3 = Down(ch3, ch3, time_emb_dim)
        self.mid1 = ResBlock(ch3, ch3, time_emb_dim)
        self.mid2 = ResBlock(ch3, ch3, time_emb_dim)
        self.up1 = Up(ch3, ch3, ch3, time_emb_dim)
        self.up2 = Up(ch3, ch3, ch2, time_emb_dim)
        self.up3 = Up(ch2, ch2, ch1, time_emb_dim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch1),
            nn.SiLU(),
            nn.Conv2d(ch1, in_ch, 1)
        )
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        x = self.up1(x, s3, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up3(x, s1, t_emb)
        return self.out_conv(x)

# ── Setup ────────────────────────────────────────────────────────────────────

T = 300
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# ── Load Model ───────────────────────────────────────────────────────────────

model = SimpleUNet(in_ch=3, base_ch=64, time_emb_dim=256).to(device)

ckpt_path = hf_hub_download(repo_id="rek49/diffusionmodel", filename="ddpm_final.pth")
ckpt = torch.load(ckpt_path, map_location=device)

from collections import OrderedDict
new_ckpt = OrderedDict()
for k, v in ckpt.items():
    new_ckpt[k.replace("module.", "")] = v

model.load_state_dict(new_ckpt)
model.eval()
print("Model loaded!")

# ── Sampler ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def p_sample(x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_recip_t = extract(sqrt_recip_alphas, t, x.shape)
    predicted_noise = model(x, t)
    model_mean = sqrt_recip_t * (x - betas_t * predicted_noise / sqrt_one_minus_t)
    if t_index == 0:
        return model_mean
    posterior_var_t = extract(posterior_variance, t, x.shape)
    return model_mean + torch.sqrt(posterior_var_t) * torch.randn_like(x)

@torch.no_grad()
def p_sample_loop(shape, n_intermediate=8):
    img = torch.randn(shape, device=device)
    intermediates = []
    save_interval = T // n_intermediate
    for i in reversed(range(0, T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(img, t, i)
        if i % save_interval == 0 or i == 0:
            intermediates.append(img.clone().cpu())
    return img.cpu(), intermediates

# ── Gradio Function ──────────────────────────────────────────────────────────

def generate(seed):
    torch.manual_seed(int(seed))
    final_img, intermediates = p_sample_loop((1, 3, IMG_SIZE, IMG_SIZE))

    n = len(intermediates) + 1
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3))
    for idx, inter in enumerate(intermediates):
        img = (inter.squeeze().permute(1,2,0).clamp(-1,1).numpy() + 1) / 2
        axes[idx].imshow(img)
        axes[idx].set_title(f"t={T - idx*(T//len(intermediates))}", fontsize=8)
        axes[idx].axis('off')

    final_np = (final_img.squeeze().permute(1,2,0).clamp(-1,1).numpy() + 1) / 2
    axes[-1].imshow(final_np)
    axes[-1].set_title("Final", fontsize=8, color='green')
    axes[-1].axis('off')

    plt.suptitle("Denoising Steps → Generated Image", fontsize=11)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, dpi=120, bbox_inches='tight')
    plt.close()
    return tmp.name, final_np

# ── Interface ────────────────────────────────────────────────────────────────

with gr.Blocks(title="DDPM Face Generator") as demo:
    gr.Markdown("# DDPM Face Generator\nGenerates faces from pure noise using a diffusion model trained on CelebA-HQ.")
    with gr.Row():
        seed_slider = gr.Slider(0, 1000, value=42, step=1, label="Random Seed")
        gen_btn = gr.Button("Generate Image", variant="primary")
    with gr.Row():
        steps_output = gr.Image(label="Denoising Steps")
        final_output = gr.Image(label="Final Generated Image")
    gen_btn.click(fn=generate, inputs=[seed_slider], outputs=[steps_output, final_output])

demo.launch()