import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAdaptiveHyperedgeGenerator(nn.Module):
    """
    Args:
        embed_dim (int): Feature dimension
        num_hyperedges (int): Number of learnable hyperedges (default: 8)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
        context (str): Context type - "both", "avg", or "max" (default: "both")
        use_cross_modal (bool): Enable cross-modal context injection (default: True)
    """
    def __init__(self, embed_dim, num_hyperedges=8, num_heads=4,
                 dropout=0.1, context="both", use_cross_modal=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.context = context
        self.use_cross_modal = use_cross_modal

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # Global prototypes (learnable)
        self.global_prototype = nn.Parameter(
            torch.randn(num_hyperedges, embed_dim)
        )
        nn.init.xavier_uniform_(self.global_prototype)

        # Context dimension calculation
        if context == "both":
            self_context_dim = embed_dim * 2  # avg + max from self
        else:
            self_context_dim = embed_dim  # only avg or max from self

        # Cross-modal context dimension (NEW vs YOLOv13)
        if use_cross_modal:
            if context == "both":
                cross_context_dim = embed_dim * 2  # avg + max from complementary modality
            else:
                cross_context_dim = embed_dim
            total_context_dim = self_context_dim + cross_context_dim

            # Cross-modal context projection (NEW)
            self.cross_modal_proj = nn.Linear(
                total_context_dim, self_context_dim
            )
        else:
            total_context_dim = self_context_dim

        # Dynamic offset generator
        self.offset_generator = nn.Linear(
            self_context_dim,  # After cross-modal projection
            num_hyperedges * embed_dim
        )

        # Query projection
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_complementary=None):
        """
        Args:
            x: Tensor (B, N, C) - Node features (current modality)
            x_complementary: Tensor (B, N, C) - Node features (complementary modality)
                             None for single-modality mode (fallback to YOLOv13)

        Returns:
            A: Tensor (B, N, M) - Participation matrix (continuous values)
        """
        B, N, C = x.shape
        M = self.num_hyperedges

        # ====================================================================
        # Step 1: Generate Global Context (ENHANCED with cross-modal)
        # ====================================================================
        # Self-modality context
        if self.context == "avg":
            self_ctx = x.mean(dim=1)  # (B, C)
        elif self.context == "max":
            self_ctx = x.max(dim=1)[0]  # (B, C)
        else:  # "both"
            avg_ctx = x.mean(dim=1)
            max_ctx = x.max(dim=1)[0]
            self_ctx = torch.cat([avg_ctx, max_ctx], dim=-1)  # (B, 2C)

        # Cross-modal context injection
        if self.use_cross_modal and x_complementary is not None:
            if self.context == "avg":
                cross_ctx = x_complementary.mean(dim=1)  # (B, C)
            elif self.context == "max":
                cross_ctx = x_complementary.max(dim=1)[0]  # (B, C)
            else:  # "both"
                avg_ctx_cross = x_complementary.mean(dim=1)
                max_ctx_cross = x_complementary.max(dim=1)[0]
                cross_ctx = torch.cat([avg_ctx_cross, max_ctx_cross], dim=-1)  # (B, 2C)

            # Fuse self and cross-modal context
            global_ctx = self.cross_modal_proj(
                torch.cat([self_ctx, cross_ctx], dim=-1)
            )  # (B, context_dim)
        else:
            # Fallback to single-modality
            global_ctx = self_ctx

        # ====================================================================
        # Step 2: Generate Dynamic Offsets
        # ====================================================================
        delta_P = self.offset_generator(global_ctx)  # (B, M*C)
        delta_P = delta_P.view(B, M, C)  # (B, M, C)

        # ====================================================================
        # Step 3: Compute Dynamic Prototypes
        # ====================================================================
        P = self.global_prototype.unsqueeze(0) + delta_P  # (B, M, C)

        # ====================================================================
        # Step 4: Generate Query Vectors
        # ====================================================================
        Q = self.query_proj(x)  # (B, N, C)

        # ====================================================================
        # Step 5: Multi-Head Attention for Similarity Computation
        # ====================================================================
        # Reshape for multi-head
        Q = Q.view(B, N, self.num_heads, self.head_dim)  # (B, N, h, d)
        P = P.view(B, M, self.num_heads, self.head_dim)  # (B, M, h, d)

        # Compute per-head similarity
        similarity = torch.einsum('bnhd,bmhd->bhnm', Q, P)  # (B, h, N, M)
        similarity = similarity / math.sqrt(self.head_dim)

        # Average across heads
        similarity = similarity.mean(dim=1)  # (B, N, M)

        # ====================================================================
        # Step 6: Softmax to Generate Participation Matrix (Continuous!)
        # ====================================================================
        A = F.softmax(similarity, dim=-1)  # (B, N, M)
        A = self.dropout(A)

        return A


class AdaptiveHypergraphConv(nn.Module):
    """
    Args:
        in_channels (int): Input channel dimension
        out_channels (int): Output channel dimension
        dropout (float): Dropout rate (default: 0.1)
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # V→E projection
        self.vertex_to_edge = nn.Linear(in_channels, out_channels)

        # E→V projection
        self.edge_to_vertex = nn.Linear(out_channels, out_channels)

        # Normalization
        self.norm = nn.LayerNorm(out_channels)

        # Activation
        self.act = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        """
        Args:
            x: Tensor (B, N, C_in) - Vertex features
            A: Tensor (B, N, M) - Participation matrix

        Returns:
            x_out: Tensor (B, N, C_out) - Updated vertex features
        """
        B, N, C_in = x.shape
        M = A.shape[2]

        # ====================================================================
        # Stage 1: V→E (Aggregate vertices to hyperedges)
        # ====================================================================
        # Project vertex features
        x_proj = self.vertex_to_edge(x)  # (B, N, C_out)

        # Aggregate: f_m = Σ(A[i,m] * x_i)
        edge_features = torch.einsum('bnm,bnc->bmc', A, x_proj)  # (B, M, C_out)

        # ====================================================================
        # Stage 2: E→V (Distribute hyperedges to vertices)
        # ====================================================================
        # Project hyperedge features
        edge_features = self.edge_to_vertex(edge_features)  # (B, M, C_out)

        # Distribute: x'_i = Σ(A[i,m] * f_m)
        x_out = torch.einsum('bnm,bmc->bnc', A, edge_features)  # (B, N, C_out)

        # ====================================================================
        # Normalization + Activation + Dropout
        # ====================================================================
        x_out = self.norm(x_out)
        x_out = self.act(x_out)
        x_out = self.dropout(x_out)

        return x_out


class EnhancedGatedResidual(nn.Module):
    """
    Args:
        channels (int): Number of channels
    """
    def __init__(self, channels):
        super().__init__()
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_hyper, x_identity):
        """
        Args:
            x_hyper: Hypergraph-enhanced features [B, C, H, W]
            x_identity: Identity features [B, C, H, W]

        Returns:
            x_out: Gated fusion result [B, C, H, W]
        """
        gate = self.gate_generator(x_hyper)  # [B, C, H, W]
        x_out = gate * x_hyper + (1 - gate) * x_identity
        return x_out


class CAH(nn.Module):
    """
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        num_hyperedges (int): Number of hyperedges (default: 8, range: 4-12)
        reduction (int): Dimensionality reduction ratio (default: 2)
        use_gate (bool): Enable gated residual fusion (default: True)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
        context (str): Context type - "both", "avg", or "max" (default: "both")
        use_cross_modal (bool): Enable cross-modal hyperedge injection (default: True)
    """
    def __init__(self, c1, c2, num_hyperedges=8, reduction=2,
                 use_gate=True, num_heads=4, dropout=0.1, context="both",
                 use_cross_modal=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.num_hyperedges = num_hyperedges
        self.c_hidden = c1 // reduction
        self.use_gate = use_gate
        self.use_cross_modal = use_cross_modal

        # Ensure hidden dimension is divisible by num_heads
        if self.c_hidden % num_heads != 0:
            self.c_hidden = (self.c_hidden // num_heads) * num_heads

        # Storage for intermediate results (for visualization)
        self.adjacency = None
        self.hyper_feat = None

        # ====================================================================
        # 1. Feature Projection (dimensionality reduction)
        # ====================================================================
        self.feature_projection = nn.Linear(c1, self.c_hidden)

        # ====================================================================
        # 2. Cross-Modal Adaptive Hyperedge Generator
        # ====================================================================
        self.hyperedge_generator = CrossModalAdaptiveHyperedgeGenerator(
            embed_dim=self.c_hidden,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            context=context,
            use_cross_modal=use_cross_modal
        )

        # ====================================================================
        # 3. Hypergraph Convolution
        # ====================================================================
        self.hypergraph_conv = AdaptiveHypergraphConv(
            in_channels=self.c_hidden,
            out_channels=self.c_hidden,
            dropout=dropout
        )

        # ====================================================================
        # 4. Feature Recovery (dimensionality expansion)
        # ====================================================================
        self.feature_recovery = nn.Linear(self.c_hidden, c2)

        # ====================================================================
        # 5. Normalization and Activation
        # ====================================================================
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        # ====================================================================
        # 6. Identity Projection (for residual connection)
        # ====================================================================
        if c1 != c2:
            self.identity_proj = nn.Conv2d(c1, c2, kernel_size=1, bias=False)
        else:
            self.identity_proj = nn.Identity()

        # ====================================================================
        # 7. Enhanced Gated Residual Fusion
        # ====================================================================
        if self.use_gate:
            self.gated_fusion = EnhancedGatedResidual(c2)

    def forward(self, x, x_complementary=None):
        """
        Forward pass with optional cross-modal context

        Args:
            x: Tensor (B, C1, H, W) - Input feature map (current modality)
            x_complementary: Tensor (B, C1, H, W) - Complementary modality features
                            (optional, for cross-modal hyperedge injection)

        Returns:
            x_out: Tensor (B, C2, H, W) - Output feature map
        """
        b, c, h, w = x.shape
        n = h * w

        # Save identity
        identity = self.identity_proj(x)  # (B, C2, H, W)

        # ====================================================================
        # Step 1: Reshape to node representation
        # ====================================================================
        x_nodes = x.view(b, c, n).transpose(1, 2).contiguous()  # (B, N, C1)

        # Complementary modality nodes (if provided)
        if x_complementary is not None:
            x_comp_nodes = x_complementary.view(b, c, n).transpose(1, 2).contiguous()  # (B, N, C1)
        else:
            x_comp_nodes = None

        # ====================================================================
        # Step 2: Feature Dimensionality Reduction
        # ====================================================================
        x_reduced = self.feature_projection(x_nodes)  # (B, N, C_hidden)

        if x_comp_nodes is not None:
            x_comp_reduced = self.feature_projection(x_comp_nodes)  # (B, N, C_hidden)
        else:
            x_comp_reduced = None

        # ====================================================================
        # Step 3: Cross-Modal Adaptive Hyperedge Generation
        # ====================================================================
        A = self.hyperedge_generator(x_reduced, x_comp_reduced)  # (B, N, M)

        # Save participation matrix for visualization
        self.adjacency = A.detach()

        # ====================================================================
        # Step 4: Hypergraph Convolution
        # ====================================================================
        x_hyper = self.hypergraph_conv(x_reduced, A)  # (B, N, C_hidden)

        # Save hypergraph features for visualization
        self.hyper_feat = x_hyper.detach()

        # ====================================================================
        # Step 5: Feature Recovery (dimension expansion to C2)
        # ====================================================================
        x_hyper = self.feature_recovery(x_hyper)  # (B, N, C2)

        # ====================================================================
        # Step 6: Reshape back to spatial dimensions
        # ====================================================================
        x_hyper = x_hyper.transpose(1, 2).contiguous().view(b, self.c2, h, w)

        # ====================================================================
        # Step 7: Normalization and Activation
        # ====================================================================
        x_hyper = self.act(self.bn(x_hyper))

        # ====================================================================
        # Step 8: Enhanced Gated Residual Fusion
        # ====================================================================
        if self.use_gate:
            x_out = self.gated_fusion(x_hyper, identity)
        else:
            x_out = x_hyper + identity

        return x_out
