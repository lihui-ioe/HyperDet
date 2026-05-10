import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class ModalityAdaptiveChannelAttention(nn.Module):
    """
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for bottleneck (default: 8)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Modality-specific FC branches
        # RGB branch: models dense texture channel correlations
        self.fc_rgb = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # IR branch: models sparse thermal channel patterns
        self.fc_ir = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # Cross-modal calibration layer
        # Jointly refines RGB and IR channel weights using complementary information
        self.calibration = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, vis_feat, ir_feat):
        """
        Args:
            vis_feat: RGB feature [B, C, H, W]
            ir_feat: IR feature [B, C, H, W]

        Returns:
            w_vis_calibrated: Calibrated RGB channel weights [B, C, 1, 1]
            w_ir_calibrated: Calibrated IR channel weights [B, C, 1, 1]
        """
        b, c, _, _ = vis_feat.size()

        # Step 1: Extract global channel statistics
        w_vis_global = self.avg_pool(vis_feat).view(b, c)  # [B, C]
        w_ir_global = self.avg_pool(ir_feat).view(b, c)    # [B, C]

        # Step 2: Modality-specific channel importance learning
        w_vis_specific = self.fc_rgb(w_vis_global)  # [B, C] - RGB-specific weights
        w_ir_specific = self.fc_ir(w_ir_global)     # [B, C] - IR-specific weights

        # Step 3: Cross-modal calibration
        # Concatenate RGB and IR weights to enable cross-modal refinement
        w_vis_calibrated = self.calibration(
            torch.cat([w_vis_specific, w_ir_specific], dim=1)
        ).view(b, c, 1, 1)  # [B, C, 1, 1]

        w_ir_calibrated = self.calibration(
            torch.cat([w_ir_specific, w_vis_specific], dim=1)
        ).view(b, c, 1, 1)  # [B, C, 1, 1]

        return w_vis_calibrated, w_ir_calibrated


class DCFA(nn.Module):
    """
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction ratio (default: 8)
        use_spatial (bool): Whether to use spatial attention (default: True)
        kernel_size (int): Base kernel size for pyramid convolutions (default: 80)
        p_kernel (list): Pyramid kernel sizes [k1, k2] (default: [5, 4])
        m_kernel (list): Multi-scale spatial kernel sizes (default: [3, 7])
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        use_spatial: bool = True,
        kernel_size: int = 80,
        p_kernel: list = None,
        m_kernel: list = None
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.use_spatial = use_spatial
        self.kernel_size = kernel_size

        # =====================================================================
        # Enhanced Channel Attention Branch
        # =====================================================================
        self.channel_attention = ModalityAdaptiveChannelAttention(channels, reduction)

        # Compression convolution: fuse RGB and IR features (2C -> C)
        self.compress = Conv(channels * 2, channels, 3)

        # Pyramid convolutions for multi-scale global context
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel

        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel1, kernel1, 0, groups=channels),
            nn.SiLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel2, kernel2, 0, groups=channels),
            nn.SiLU()
        )
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                int(self.kernel_size / kernel1 / kernel2),
                int(self.kernel_size / kernel1 / kernel2),
                0,
                groups=channels
            ),
            nn.SiLU()
        )

        # =====================================================================
        # Spatial Attention Branch
        # =====================================================================
        if self.use_spatial:
            if m_kernel is None:
                m_kernel = [3, 7]

            # Multi-scale spatial convolutions for RGB
            self.cv_v1 = Conv(channels, 1, m_kernel[0])
            self.cv_v2 = Conv(channels, 1, m_kernel[1])

            # Multi-scale spatial convolutions for IR
            self.cv_i1 = Conv(channels, 1, m_kernel[0])
            self.cv_i2 = Conv(channels, 1, m_kernel[1])

            # Spatial weight fusion convolutions
            self.conv1 = Conv(2, 1, 5)
            self.conv2 = Conv(2, 1, 5)

            # Global spatial feature compression
            self.compress1 = Conv(channels, 1, 3)
            self.compress2 = Conv(channels, 1, 3)

        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, data):
        """
        Args:
            data: tuple/list of (vis_feat, ir_feat)
                vis_feat: RGB features [B, C, H, W]
                ir_feat: IR features [B, C, H, W]

        Returns:
            tuple: (vis_enhanced, ir_enhanced)
                vis_enhanced: Enhanced RGB features [B, C, H, W]
                ir_enhanced: Enhanced IR features [B, C, H, W]
        """
        vis_feat = data[0]
        ir_feat = data[1]

        b, c, h, w = vis_feat.size()

        # =====================================================================
        # Stage 1: Modality-Adaptive Channel Attention
        # =====================================================================

        # 1.1 Extract modality-specific channel weights with cross-modal calibration
        w_vis, w_ir = self.channel_attention(vis_feat, ir_feat)  # [B, C, 1, 1] each

        # 1.2 Global context extraction from fused features
        glob_t = self.compress(torch.cat([vis_feat, ir_feat], 1))  # [B, C, H, W]

        # 1.3 Pyramid convolution or global pooling for multi-scale context
        if min(h, w) >= self.kernel_size:
            # Use pyramid convolution for large feature maps
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))  # [B, C, H', W']
            # Ensure 1x1 output
            glob = torch.mean(glob, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        else:
            # Use global pooling for small feature maps
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # 1.4 Cross-modal channel enhancement
        # Key: RGB uses IR weights, IR uses RGB weights
        result_vis_ca = vis_feat * (self.act(w_ir * glob)).expand_as(vis_feat)
        result_ir_ca = ir_feat * (self.act(w_vis * glob)).expand_as(ir_feat)

        # =====================================================================
        # Stage 2: Spatial Attention
        # =====================================================================
        if self.use_spatial:
            # 2.1 Multi-scale spatial feature extraction
            w_vis_sp = self.conv1(torch.cat([
                self.cv_v1(result_vis_ca),  # Small receptive field
                self.cv_v2(result_vis_ca)   # Large receptive field
            ], 1))  # [B, 1, H, W]

            w_ir_sp = self.conv2(torch.cat([
                self.cv_i1(result_ir_ca),
                self.cv_i2(result_ir_ca)
            ], 1))  # [B, 1, H, W]

            # 2.2 Global spatial context
            glob_sp = self.act(
                self.compress1(result_vis_ca) + self.compress2(result_ir_ca)
            )  # [B, 1, H, W]

            # 2.3 Spatial weight fusion
            w_vis_sp = self.act(glob_sp + w_vis_sp)  # [B, 1, H, W]
            w_ir_sp = self.act(glob_sp + w_ir_sp)   # [B, 1, H, W]

            # 2.4 Cross-modal spatial enhancement
            result_vis = result_vis_ca * w_ir_sp.expand_as(result_vis_ca)
            result_ir = result_ir_ca * w_vis_sp.expand_as(result_ir_ca)
        else:
            result_vis = result_vis_ca
            result_ir = result_ir_ca

        # =====================================================================
        # Stage 3: Residual Fusion with Activation
        # =====================================================================
        result_vis = self.act(result_vis + vis_feat)
        result_ir = self.act(result_ir + ir_feat)

        return result_vis, result_ir
    
