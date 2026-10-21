T = 1000
s = 8e-3

# Standard cosine beta schedule (corrected step division)
steps = torch.linspace(
    0, T, T + 1,
    dtype=torch.float64,
    device=device
)
alpha_bar = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
alpha_bar = alpha_bar / alpha_bar[0]

betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
betas = betas.clamp(min=1e-5, max=0.999).float()

alphas = 1.0 - betas
alpha_time_t = torch.cumprod(alphas, dim=0)

from google.colab import drive
drive.mount('/content/gdrive')
PATH = "/content/gdrive/My Drive/ddpm_cifar.pt"
torch.save(model.state_dict(), PATH)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, time_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        #initial convolution to get 64 channels from the base 1 channel since the pictures are greyscale
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        #first downblock, convoluting and learning features while sending skip connection to upblocks
        self.downblock1 = resnetblock(base_channels, base_channels, time_dim)

        #first downsample, where number of channels increases for more feature learning and stride=2 to turn the 28 by 28 pictures into 14 by 14
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size = 4, stride=2, padding=1)

        self.downblock2 = resnetblock(base_channels * 2, base_channels * 2, time_dim)

        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.downblock3 = resnetblock(base_channels * 4, base_channels * 4, time_dim)

        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)

        self.bottleneck1 = resnetblock(base_channels * 8, base_channels * 8, time_dim)

        self.bottleneck2 = resnetblock(base_channels * 8, base_channels * 8, time_dim)

        self.bottleneck3 = resnetblock(base_channels * 8, base_channels * 8, time_dim)

        #first upsample, reducing number of channels and increasing from 7 by 7 image to 14 by 14 using convtranspose2d
        self.upsample1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size = 4, stride=2, padding=1)

        # Fix: input channels for upblock1 should be base_channels * 4 due to concatenation
        self.upblock1 = resnetblock(base_channels * 8, base_channels * 4, time_dim)

        self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # Fix: input channels for upblock2 should be base_channels * 2 due to concatenation
        self.upblock2 = resnetblock(base_channels * 4, base_channels * 2, time_dim)

        self.upsample3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)

        self.upblock3 = resnetblock(base_channels * 2, base_channels, time_dim)

        self.output = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = self.input_conv(x)
        d1 = self.downblock1(x, t_emb)
        x = self.downsample1(d1)
        d2 = self.downblock2(x, t_emb)
        x = self.downsample2(d2)
        d3 = self.downblock3(x, t_emb)
        x = self.downsample3(d3)
        x = self.bottleneck1(x, t_emb)
        x = self.bottleneck2(x, t_emb)
        x = self.bottleneck3(x, t_emb)
        x = self.upsample1(x)
        x = torch.cat([x, d3], dim=1)
        u1 = self.upblock1(x, t_emb)
        x = self.upsample2(u1)
        x = torch.cat([x, d2], dim=1)
        u2 = self.upblock2(x, t_emb)
        x = self.upsample3(u2)
        x = torch.cat([x, d1], dim=1)
        u3 = self.upblock3(x, t_emb)
        return self.output(u3)
