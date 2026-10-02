"""
Model components for latent world models.

Inherited from Le-WM (https://github.com/lucas-maes/le-wm) with the addition of InverseModel.
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class PredictiveMaskPolicy(nn.Module):
    """State-only fixed-budget patch masking policy.

    The forward pass returns an exactly-K hard mask, while its backward pass
    follows a temperature-smoothed sigmoid mask.  ``base_scores`` are a fixed
    tie-breaker (and intentionally are not a parameter); the learned MLP is
    therefore precisely the residual score term in the experiment design.
    """

    def __init__(
        self,
        input_dim: int,
        num_patches: int,
        hidden_dim: int = 256,
        alpha: float = 1.0,
        temperature: float = 0.5,
        base_seed: int = 0,
    ):
        super().__init__()
        if int(num_patches) < 2:
            raise ValueError("PredictiveMaskPolicy requires at least two patches")
        if float(temperature) <= 0:
            raise ValueError("mask-policy temperature must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(base_seed))


        self.register_buffer(
            "base_scores", torch.rand(int(num_patches), generator=generator) - 0.5
        )
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.num_patches = int(num_patches)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_patches)),
        )


        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def scores(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.base_scores.to(state_features.dtype) + self.alpha * self.net(state_features)

    def forward(self, state_features: torch.Tensor, num_masked: int):
        """Return ``(hard_mask, straight_through_mask, soft_mask, scores)``.

        ``state_features`` has arbitrary leading dimensions ending in the
        encoder feature dimension. The final output dimension is the ViT patch
        index. A differentiable bisection solves the sigmoid budget threshold;
        hard TopK is used only in the forward value.
        """
        if not 0 < int(num_masked) < self.num_patches:
            raise ValueError(
                f"num_masked must be in (0, {self.num_patches}), got {num_masked}"
            )


        scores = self.scores(state_features).float()
        hard = torch.zeros_like(scores)
        indices = scores.topk(int(num_masked), dim=-1).indices
        hard.scatter_(-1, indices, 1.0)




        lower = scores.detach().amin(dim=-1, keepdim=True) - 20.0 * self.temperature
        upper = scores.detach().amax(dim=-1, keepdim=True) + 20.0 * self.temperature
        target = float(num_masked)
        for _ in range(32):
            midpoint = (lower + upper) * 0.5
            mass = torch.sigmoid((scores - midpoint) / self.temperature).sum(dim=-1, keepdim=True)
            lower = torch.where(mass.detach() > target, midpoint, lower)
            upper = torch.where(mass.detach() > target, upper, midpoint)
        kappa = (lower + upper) * 0.5
        soft = torch.sigmoid((scores - kappa) / self.temperature)
        straight_through = soft + (hard - soft).detach()
        return hard, straight_through, soft, scores


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class _ActionInnovationGradient(torch.autograd.Function):
    """Identity in the forward pass, factual-minus-reference in backward.

    For action conditioner outputs ``c(a)`` and ``c(a_bar)``, this is the
    exact-forward implementation of ``c(a) - c(a_bar) + sg[c(a_bar)]``.  A
    literal subtraction/addition is mathematically equivalent but can change
    low-precision forward rounding; returning ``factual`` directly preserves
    an existing LeWM checkpoint's forward function bit-for-bit.
    """

    @staticmethod
    def forward(ctx, factual, reference):
        return factual

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output


def action_innovation_gradient(factual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Apply the AIG gradient replacement while retaining ``factual`` forward."""
    return _ActionInnovationGradient.apply(factual, reference)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, c_reference=None, use_aig=False):
        modulation = self.adaLN_modulation(c)
        if use_aig:
            if c_reference is None:
                raise ValueError("AIG requires a reference action conditioner")
            modulation = action_innovation_gradient(
                modulation, self.adaLN_modulation(c_reference)
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=ConditionalBlock,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward_features(self, x, c=None, c_reference=None, use_aig=False):
        """Return the final hidden tokens, before the output projection.

        This is intentionally a separate method rather than a hook: PC-WM
        needs the action-conditioned feature used by the predictor while the
        ordinary forward path must remain numerically unchanged.
        """
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)
            if c_reference is not None:
                c_reference = self.cond_proj(c_reference)
        for block in self.layers:
            x = block(x, c, c_reference=c_reference, use_aig=use_aig)
        return self.norm(x)

    def forward(self, x, c=None, c_reference=None, use_aig=False):
        x = self.forward_features(x, c, c_reference=c_reference, use_aig=use_aig)
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim=None,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        aig_enabled=False,
        aig_behavior_hidden_dim=256,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.aig_enabled = bool(aig_enabled)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )



        if self.aig_enabled:
            if action_dim is None or int(action_dim) <= 0:
                raise ValueError("AIG ARPredictor requires a positive action_dim")
            self.behavior_head = nn.Sequential(
                nn.Linear(input_dim, int(aig_behavior_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(aig_behavior_hidden_dim), int(action_dim)),
            )

    def history_features(self, x):
        """State/history token before any action is injected into AdaLN."""
        T = x.size(1)
        return x + self.pos_embedding[:, :T]

    def behavior_mean_from_features(self, h, detach_input=True):
        if not self.aig_enabled:
            raise RuntimeError("behavior_mean_from_features requires aig_enabled=true")
        return self.behavior_head(h.detach() if detach_input else h)

    def behavior_loss_from_features(self, h, action):
        return (self.behavior_mean_from_features(h, detach_input=True) - action).pow(2).mean()

    def forward(self, x, c, c_reference=None, use_aig=False):
        if use_aig and not self.aig_enabled:
            raise RuntimeError("use_aig=True requires predictor.aig_enabled=true")
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c, c_reference=c_reference, use_aig=use_aig)
        return x

    def forward_with_features(self, x, c):
        """Return prediction tokens and action-conditioned hidden tokens.

        PC-WM consumes ``hidden`` with stop-gradient.  This method is only
        used after the world model is frozen, so it deliberately has no AIG
        branch and leaves the standard ``forward`` implementation untouched.
        """
        T = x.size(1)
        tokens = self.dropout(x + self.pos_embedding[:, :T])
        hidden = self.transformer.forward_features(tokens, c)
        prediction = self.transformer.output_proj(hidden)
        return prediction, hidden


class PlainBlock(nn.Module):
    """Pre-norm transformer block with no external conditioning.

    Used for the AFT trunk: unlike `ConditionalBlock`, it must not see the
    action, since its output `h` is the basis for the action-free drift term.
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, c=None, c_reference=None, use_aig=False):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class EndpointConditionalBlock(ConditionalBlock):
    """AdaLN block for masked endpoint completion.

    A causal attention mask would prevent the first slot from reading the
    second one, making reverse completion impossible.  The two positions are
    endpoint slots rather than an autoregressive sequence, so attention here
    is deliberately bidirectional.  Direction is represented by the masked
    slot's position, not by a second predictor or a sign-flipped action.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)





        nn.init.normal_(self.adaLN_modulation[-1].weight, std=0.02)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, c_reference=None, use_aig=False):
        modulation = self.adaLN_modulation(c)
        if use_aig:
            if c_reference is None:
                raise ValueError("AIG requires a reference action conditioner")
            modulation = action_innovation_gradient(
                modulation, self.adaLN_modulation(c_reference)
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), causal=False
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class RelationBlock(PlainBlock):
    """Bidirectional self-attention block for masked transition relations.

    The three slots are a set of labelled transition variables rather than an
    autoregressive sequence.  Each masked query must be able to attend to the
    two visible variables, hence deliberately non-causal attention.
    """

    def forward(self, x, c=None, c_reference=None, use_aig=False):
        x = x + self.attn(x, causal=False)
        x = x + self.mlp(x)
        return x


class MaskedTransitionPredictor(nn.Module):
    """One predictor for two queries on an action-labelled transition.

    A transition is represented by the three typed slots ``[z_t,z_{t+1},a]``.
    The same bidirectional Transformer is used to complete either a masked
    future state or a masked action.  The two output projections only decode
    the corresponding target spaces; they are not separate predictors.

    This first version intentionally models exactly one endpoint transition
    at a time.  It is appropriate for Reacher's single action chunk, whereas
    deterministic recovery of a long action chunk from only two endpoints is
    generally non-identifiable.
    """

    is_masked_transition = True

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        if int(num_frames) != 1:
            raise ValueError(
                "MaskedTransitionPredictor currently represents one transition "
                f"at a time; got num_frames={num_frames}"
            )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.action_dim = int(action_dim)
        self.action_in = nn.Linear(self.action_dim, self.input_dim)
        self.future_mask = nn.Parameter(torch.randn(1, 1, self.input_dim) * 0.02)
        self.action_mask = nn.Parameter(torch.randn(1, 1, self.input_dim) * 0.02)

        self.type_embedding = nn.Parameter(torch.randn(1, 3, self.input_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            self.input_dim, hidden_dim, hidden_dim, depth, heads, dim_head,
            mlp_dim, dropout, block_class=RelationBlock,
        )
        self.state_out = nn.Linear(hidden_dim, self.output_dim)
        self.action_out = nn.Linear(hidden_dim, self.action_dim)

    def _run(self, z_t, future_slot, action_slot):
        if z_t.ndim != 3 or z_t.size(1) != 1:
            raise ValueError(
                "MaskedTransitionPredictor expects z_t shaped [B,1,D]"
            )
        b = z_t.size(0)
        slots = torch.cat([z_t, future_slot, action_slot], dim=1)
        if slots.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected latent dim {self.input_dim}, got {slots.shape[-1]}"
            )
        return self.transformer(self.dropout(slots + self.type_embedding))

    def complete_future(self, z_t, action):
        """``[z_t, M_z, a] -> z_{t+1}``, used for learning and planning."""
        b = z_t.size(0)
        future_mask = self.future_mask.expand(b, -1, -1)
        action_slot = self.action_in(action)
        return self.state_out(self._run(z_t, future_mask, action_slot)[:, 1:2])

    def complete_action(self, z_t, z_next):
        """``[z_t, z_{t+1}, M_a] -> a``; caller controls endpoint gradients."""
        b = z_t.size(0)
        action_mask = self.action_mask.expand(b, -1, -1)
        return self.action_out(self._run(z_t, z_next, action_mask)[:, 2:3])

    def forward(self, x, action, c_reference=None, use_aig=False):

        if c_reference is not None or use_aig:
            raise ValueError("AIG is only defined for the original AR AdaLN predictor")
        return self.complete_future(x, action)


def _orthonormal_frame(M: torch.Tensor) -> torch.Tensor:
    """QR-orthonormalize the columns of M: (..., D, m) -> Q with Q^T Q = I_m."""
    dtype = M.dtype
    Q, _ = torch.linalg.qr(M.float(), mode="reduced")
    return Q.to(dtype)


class AFTPredictor(nn.Module):
    """Action-Faithful Transport (AFT) predictor.

        T(z, a) = b(z) + Q(z) a + P_Q(z)^perp [r(z, a) - r(z, 0)]

    `Q(z)` is an orthonormal action tangent frame (Q^T Q = I_m, m = action
    dim), obtained by QR-orthonormalizing a learned `D x m` matrix. This makes
    the action term identifiable and non-cancellable:

        Q(z)^T [T(z, a) - T(z, 0)] = a   identically,

    since Q^T Q = I and Q^T P_Q^perp = 0. A constant encoder can therefore no
    longer drive the prediction loss to zero whenever actions have non-zero
    variance, without any statistical (SIGReg/VICReg) regularizer or auxiliary
    anti-collapse head. `b(z) = T(z, 0)` is the strongest action-free
    shortcut the trunk can extract; `r(z, a) - r(z, 0)` is confined to the
    orthogonal complement of the action frame, so it can express contact/
    interaction dynamics without being able to absorb `Qa`.
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim or input_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.trunk = Transformer(
            input_dim,
            hidden_dim,
            hidden_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=PlainBlock,
        )
        self.drift_head = nn.Linear(hidden_dim, self.output_dim)
        self.tangent_head = nn.Linear(hidden_dim, self.output_dim * self.action_dim)
        self.residual_head = MLP(hidden_dim + self.action_dim, hidden_dim, self.output_dim)

        nn.init.zeros_(self.drift_head.bias)
        nn.init.zeros_(self.tangent_head.bias)

    def forward(self, x, action):
        """
        x:      (B, T, input_dim)  - history of latent states z_t
        action: (B, T, action_dim) - raw (unembedded) action at each z_t
        Returns predicted next-state embedding, (B, T, output_dim).
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        h = self.trunk(x)

        b = self.drift_head(h)
        M = self.tangent_head(h).unflatten(-1, (self.output_dim, self.action_dim))
        Q = _orthonormal_frame(M)

        zero_action = torch.zeros_like(action)
        delta_r = self.residual_head(torch.cat([h, action], dim=-1)) - self.residual_head(
            torch.cat([h, zero_action], dim=-1)
        )

        coeff = torch.einsum("...dm,...d->...m", Q, delta_r)
        delta_r_perp = delta_r - torch.einsum("...dm,...m->...d", Q, coeff)
        action_transport = torch.einsum("...dm,...m->...d", Q, action)

        return b + action_transport + delta_r_perp


class CAFEPredictor(nn.Module):
    """Causal Action Free Energy (CAFE) control-affine predictor.

    ``P(z, a) = b(z) + C(z) a`` with Causal Capacity Normalization (CCN)
    built into the parameterization: ``||C(z)||_F^2 = action_dim`` except at
    the numerical zero-norm guard. It consumes normalized raw actions.
    """

    def __init__(
        self, *, num_frames, depth, heads, mlp_dim, input_dim, hidden_dim,
        action_dim, output_dim=None, dim_head=64, dropout=0.0,
        emb_dropout=0.0, ccn_eps=1e-8,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim or input_dim)
        self.ccn_eps = float(ccn_eps)
        if self.action_dim <= 0:
            raise ValueError("CAFEPredictor requires a positive action_dim")
        if self.action_dim > self.output_dim:
            raise ValueError(
                "CAFEPredictor requires action_dim <= output_dim; an "
                "action isometry cannot fit in a smaller latent"
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.trunk = Transformer(
            input_dim, hidden_dim, hidden_dim, depth, heads, dim_head, mlp_dim,
            dropout, block_class=PlainBlock,
        )
        self.drift_head = nn.Linear(hidden_dim, self.output_dim)
        self.tangent_head = nn.Linear(hidden_dim, self.output_dim * self.action_dim)
        nn.init.zeros_(self.drift_head.bias)

        nn.init.normal_(self.tangent_head.weight, std=0.02)
        nn.init.normal_(self.tangent_head.bias, std=0.02)

    def causal_map(self, x):
        """Return drift ``b`` and CCN-normalized map ``C`` (..., D, m)."""
        T = x.size(1)
        h = self.trunk(self.dropout(x + self.pos_embedding[:, :T]))
        b = self.drift_head(h)
        M = self.tangent_head(h).unflatten(-1, (self.output_dim, self.action_dim))
        norm = torch.linalg.vector_norm(M.float(), dim=(-2, -1), keepdim=True)
        C = M * (self.action_dim**0.5) / norm.clamp_min(self.ccn_eps).to(M.dtype)
        return b, C

    def forward_with_causal_map(self, x, action):
        b, C = self.causal_map(x)
        return b + torch.einsum("...dm,...m->...d", C, action), C

    def forward(self, x, action):
        return self.forward_with_causal_map(x, action)[0]


class LAFPredictor(nn.Module):
    """Latent Action Field (LAF) predictor.

    The predictor never conditions its state-processing path on an action.
    Instead it maps a latent state to an orthonormal frame ``G(z)`` whose
    columns are physical-action directions in the same latent space:

        z_next = z + G(z) a.

    Datasets with action repeat / frame skip store a flattened chunk of
    physical actions.  LAF applies the *same* state-only field recursively to
    every physical action in that chunk, rather than treating the whole chunk
    as one non-identifiable endpoint action.
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        action_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        physical_action_dim=None,
        action_repeat=None,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.action_dim = int(action_dim)
        self.physical_action_dim = int(
            physical_action_dim
            if physical_action_dim is not None
            else self.action_dim
        )

        if self.num_frames != 1:
            raise ValueError(
                "LAFPredictor currently models one latent state at a time; "
                f"got num_frames={self.num_frames}"
            )
        if self.input_dim != self.output_dim:
            raise ValueError(
                "LAFPredictor is a latent displacement model and requires "
                f"input_dim == output_dim, got {self.input_dim} and {self.output_dim}"
            )
        if self.physical_action_dim <= 0 or self.physical_action_dim > self.output_dim:
            raise ValueError(
                "LAFPredictor requires 0 < physical_action_dim <= latent dimension"
            )
        if self.action_dim % self.physical_action_dim != 0:
            raise ValueError(
                "Flattened action_dim must be divisible by physical_action_dim: "
                f"got {self.action_dim} and {self.physical_action_dim}"
            )

        inferred_repeat = self.action_dim // self.physical_action_dim
        self.action_repeat = int(action_repeat or inferred_repeat)
        if self.action_repeat != inferred_repeat:
            raise ValueError(
                "action_repeat must agree with flattened action dimension: "
                f"expected {inferred_repeat}, got {self.action_repeat}"
            )


        self.trunk = Transformer(
            self.input_dim,
            hidden_dim,
            hidden_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=PlainBlock,
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.field_head = nn.Linear(
            hidden_dim, self.output_dim * self.physical_action_dim
        )

        nn.init.normal_(self.field_head.weight, std=0.02)
        nn.init.normal_(self.field_head.bias, std=0.02)

    def fields(self, z):
        """Return the state-only orthonormal action frame G(z), (..., D, m)."""
        h = self.trunk(self.dropout(z))
        M = self.field_head(h).unflatten(
            -1, (self.output_dim, self.physical_action_dim)
        )
        return _orthonormal_frame(M)

    def inverse_physical_action(self, z_t, z_tp1):
        """Analytically recover one physical action from an adjacent transition.

        For a frameskipped endpoint this is intentionally *not* an inverse of
        the entire flattened chunk: endpoint transitions do not uniquely
        identify all physical actions.  Call it on adjacent latent states.
        """
        G = self.fields(z_t)
        return torch.einsum("...dm,...d->...m", G, z_tp1 - z_t)

    def forward(self, x, action):
        """Recursively apply physical action fields for a flattened action chunk."""
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, got {action.shape[-1]}"
            )
        physical_actions = action.unflatten(
            -1, (self.action_repeat, self.physical_action_dim)
        )
        z = x
        for k in range(self.action_repeat):
            G = self.fields(z)
            dz = torch.einsum("...dm,...m->...d", G, physical_actions[..., k, :])
            z = z + dz
        return z


class EndpointCompletionPredictor(nn.Module):
    """One action-conditioned predictor for both directions of an edge.

    Each action-labelled transition is represented by two endpoint slots.  A
    known endpoint occupies one slot and a learned MASK token occupies the
    other.  The location of MASK determines whether the model completes the
    future or the predecessor; the action is injected through exactly the
    usual AdaLN path in both cases.

    ``forward=True`` means ``(z_t, MASK; a_t) -> z_{t+1}``; ``forward=False``
    means ``(MASK, z_{t+1}; a_t) -> z_t``.  Crucially, the latter uses the
    same executed action, not ``-a_t``.
    """

    is_endpoint_completion = True

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        if int(num_frames) != 1:
            raise ValueError(
                "EndpointCompletionPredictor currently expects one endpoint "
                f"per transition, got num_frames={num_frames}"
            )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim or input_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, input_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            self.output_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=EndpointConditionalBlock,
        )

    @staticmethod
    def _direction_mask(forward, known):
        """Normalize direction to bool ``(B,T)``; True selects slot 1."""
        if forward is None:
            return torch.ones(known.shape[:-1], dtype=torch.bool, device=known.device)
        if not isinstance(forward, torch.Tensor):
            return torch.full(
                known.shape[:-1], bool(forward), dtype=torch.bool, device=known.device
            )
        direction = forward.to(device=known.device, dtype=torch.bool)
        while direction.ndim < known.ndim - 1:
            direction = direction.unsqueeze(-1)
        return direction.expand(known.shape[:-1])

    def complete(self, known, action, forward=True):
        """Complete the missing transition endpoint from either edge side.

        known: ``(B,T,D)`` (for this version ``T=1``), action: AdaLN action
        embeddings with the same leading dimensions, forward: bool tensor or
        scalar.  The returned tensor has shape ``(B,T,output_dim)``.
        """
        if known.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected endpoint dimension {self.input_dim}, got {known.shape[-1]}"
            )
        if action.shape[:-1] != known.shape[:-1]:
            raise ValueError(
                "Action and endpoint leading dimensions must match: "
                f"got {tuple(action.shape)} and {tuple(known.shape)}"
            )
        direction = self._direction_mask(forward, known)
        b, t, _ = known.shape



        known_flat = known.reshape(b * t, self.input_dim)
        action_flat = action.reshape(b * t, action.shape[-1])
        direction_flat = direction.reshape(b * t)
        mask = self.mask_token.expand(b * t, -1, -1).squeeze(-2)


        slots = torch.stack((
            torch.where(direction_flat.unsqueeze(-1), known_flat, mask),
            torch.where(direction_flat.unsqueeze(-1), mask, known_flat),
        ), dim=-2)
        slots = self.dropout(slots + self.pos_embedding)
        cond = action_flat.unsqueeze(-2).expand(-1, 2, -1)
        outputs = self.transformer(slots, cond)
        output_slot = direction_flat.long().view(-1, 1, 1).expand(
            -1, 1, self.output_dim
        )
        return outputs.gather(-2, output_slot).squeeze(-2).reshape(b, t, self.output_dim)

    def forward(self, x, action):


        return self.complete(x, action, forward=True)


def build_predictor(predictor_type: str, *, num_frames, input_dim, hidden_dim, output_dim, action_dim, **kwargs):
    """Construct the transition predictor named by ``predictor_type``."""
    if predictor_type == "ar":
        return ARPredictor(
            num_frames=num_frames,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            **kwargs,
        )
    if predictor_type == "aft":
        return AFTPredictor(
            num_frames=num_frames,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            **kwargs,
        )
    if predictor_type == "cafe":
        return CAFEPredictor(
            num_frames=num_frames,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            **kwargs,
        )
    if predictor_type == "laf":
        return LAFPredictor(
            num_frames=num_frames,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            **kwargs,
        )
    if predictor_type in ("masked_transition", "mtm", "transition_jepa"):
        return MaskedTransitionPredictor(
            num_frames=num_frames,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            **kwargs,
        )
    if predictor_type in ("endpoint_completion", "edge_completion"):
        return EndpointCompletionPredictor(
            num_frames=num_frames,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs,
        )
    raise ValueError(f"Unknown predictor_type: {predictor_type!r}")


class InverseModel(nn.Module):
    """Predicts action from consecutive latent embeddings: (z_t, z_{t+1}) -> a_t"""

    def __init__(self, embed_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t, z_tp1):
        """
        z_t:   (B, D) or (B, T, D)
        z_tp1: (B, D) or (B, T, D)
        Returns: predicted action, same leading dims
        """
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


class ActionCorrector(nn.Module):
    """PC-WM's residual-to-action vector field.

    Every leading dimension is preserved, hence a Cube macro action is
    corrected as one complete token (for example all 25 coordinates) instead
    of independently correcting its physical sub-actions.
    """

    def __init__(self, action_dim, hidden_dim, latent_dim, mlp_dim=512, depth=2):
        super().__init__()
        action_dim, hidden_dim, latent_dim = map(int, (action_dim, hidden_dim, latent_dim))
        if min(action_dim, hidden_dim, latent_dim, int(mlp_dim), int(depth)) <= 0:
            raise ValueError("ActionCorrector dimensions and depth must be positive")
        layers = [nn.Linear(action_dim + hidden_dim + latent_dim, int(mlp_dim)), nn.SiLU()]
        for _ in range(int(depth) - 1):
            layers.extend([nn.Linear(int(mlp_dim), int(mlp_dim)), nn.SiLU()])
        layers.append(nn.Linear(int(mlp_dim), action_dim))
        self.net = nn.Sequential(*layers)


        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, action, hidden, residual):
        if action.shape[:-1] != hidden.shape[:-1] or action.shape[:-1] != residual.shape[:-1]:
            raise ValueError(
                "action, hidden, and residual must have identical leading dimensions"
            )
        return self.net(torch.cat((action, hidden, residual), dim=-1))


class SecantActionCorrector(nn.Module):
    """Amortized local inverse-Jacobian action update for a frozen predictor.

    The three inputs exactly match the AEM secant construction:
    ``C(z_t, a_hat, z_target - P(z_t, a_hat))``.  Its output is an additive
    action update, initialized to zero so an untrained artifact is identical
    to ordinary CEM at execution time.
    """

    def __init__(self, latent_dim, action_dim, mlp_dim=256, depth=2):
        super().__init__()
        latent_dim, action_dim, mlp_dim, depth = map(
            int, (latent_dim, action_dim, mlp_dim, depth)
        )
        if min(latent_dim, action_dim, mlp_dim, depth) <= 0:
            raise ValueError("SecantActionCorrector dimensions and depth must be positive")
        layers = [nn.Linear(2 * latent_dim + action_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, action_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, action, endpoint_residual):
        if not (z_t.shape[:-1] == action.shape[:-1] == endpoint_residual.shape[:-1]):
            raise ValueError("Secant corrector inputs must have matching leading dimensions")
        return self.net(torch.cat((z_t, action, endpoint_residual), dim=-1))


class PlanToActResidualCalibrator(nn.Module):
    """Small state-conditioned transport from model to physical actions.

    The module predicts an *additive* residual in the frozen world model's
    normalized action coordinates.  Bounding/projecting the transported
    action is deliberately the caller's responsibility: training and control
    must use the same physical action box.
    """

    def __init__(self, latent_dim, action_dim, mlp_dim=256, depth=2):
        super().__init__()
        latent_dim, action_dim, mlp_dim, depth = map(
            int, (latent_dim, action_dim, mlp_dim, depth)
        )
        if min(latent_dim, action_dim, mlp_dim, depth) <= 0:
            raise ValueError("PARC dimensions and depth must be positive")
        layers = [nn.Linear(latent_dim + action_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, action_dim))
        self.net = nn.Sequential(*layers)


        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, model_action):
        if z_t.shape[:-1] != model_action.shape[:-1]:
            raise ValueError("PARC z_t and action must have matching leading dimensions")
        return self.net(torch.cat((z_t, model_action), dim=-1))


class ActionResidualFWM(nn.Module):
    """Frozen-forward-model action-centred residual dynamics block.

    Given a frozen forward model ``P``, this module supplies only

        R(z, a) - R(z, 0)

    to its prediction.  The subtraction is structural rather than a loss
    preference: a branch which ignores its action input cancels identically.
    The final layer starts at zero, so attaching a newly created block leaves
    the base forward model exactly unchanged.
    """

    def __init__(self, embed_dim, action_dim, mlp_dim=512, depth=2,
                 condition_on_base_prediction=False):
        super().__init__()
        embed_dim, action_dim, mlp_dim, depth = map(
            int, (embed_dim, action_dim, mlp_dim, depth)
        )
        if min(embed_dim, action_dim, mlp_dim, depth) <= 0:
            raise ValueError("ActionResidualFWM dimensions and depth must be positive")
        self.embed_dim, self.action_dim = embed_dim, action_dim
        self.condition_on_base_prediction = bool(condition_on_base_prediction)
        input_dim = 2 * embed_dim + action_dim if self.condition_on_base_prediction else embed_dim + action_dim
        layers = [nn.Linear(input_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, embed_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def raw(self, z_t, action, base_prediction=None):
        if z_t.shape[:-1] != action.shape[:-1]:
            raise ValueError("z_t and action must have identical leading dimensions")
        if z_t.size(-1) != self.embed_dim or action.size(-1) != self.action_dim:
            raise ValueError(
                f"expected z/action dims {self.embed_dim}/{self.action_dim}, got "
                f"{z_t.size(-1)}/{action.size(-1)}"
            )
        features = [z_t, action]
        if self.condition_on_base_prediction:
            if base_prediction is None or base_prediction.shape != z_t.shape:
                raise ValueError("AR-FWM configured with base-prediction conditioning needs base_prediction shaped like z_t")
            features.append(base_prediction)
        return self.net(torch.cat(features, dim=-1))

    def forward(self, z_t, action, base_prediction=None):


        return self.raw(z_t, action, base_prediction) - self.raw(
            z_t, torch.zeros_like(action), base_prediction
        )


class SelfConditionedResidualFWM(nn.Module):
    """Second predictor stage conditioned on the frozen first-stage transition.

    ``base_delta`` is ``sg(P(z, a) - z)``.  Unlike the legacy post-hoc
    action-centred AR-FWM above, this is an ordinary residual *predictor*
    block: it learns a direct latent correction which is added to the first
    stage.  Its final layer is zero-initialized so inserting it initially
    preserves the exact base predictor function.
    """

    def __init__(self, embed_dim, action_dim, mlp_dim=512, depth=2):
        super().__init__()
        embed_dim, action_dim, mlp_dim, depth = map(
            int, (embed_dim, action_dim, mlp_dim, depth)
        )
        if min(embed_dim, action_dim, mlp_dim, depth) <= 0:
            raise ValueError("SelfConditionedResidualFWM dimensions and depth must be positive")
        self.embed_dim, self.action_dim = embed_dim, action_dim
        layers = [nn.Linear(2 * embed_dim + action_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, embed_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, action, base_delta):
        if z_t.shape[:-1] != action.shape[:-1] or base_delta.shape != z_t.shape:
            raise ValueError("SC-ResFWM expects z_t/base_delta=(...,D) and action=(...,A)")
        if z_t.size(-1) != self.embed_dim or action.size(-1) != self.action_dim:
            raise ValueError(
                f"expected z/action dims {self.embed_dim}/{self.action_dim}, got "
                f"{z_t.size(-1)}/{action.size(-1)}"
            )
        return self.net(torch.cat((z_t, action, base_delta), dim=-1))


class LatentFlowResidualClosure(nn.Module):
    """Action-free closure of a frozen predictor's latent vector field.

    Given the current latent state ``z_t`` and the motion proposed by the
    base predictor, ``delta_hat = P(z_t, a_t) - z_t``, the block predicts an
    additive latent correction.  It deliberately never receives an action,
    a goal, or a future/target latent, so action information can reach it
    only through the base predictor's proposed motion.

    The last layer is zero-initialized: attaching a fresh closure preserves
    the original transition exactly.
    """

    def __init__(self, embed_dim, mlp_dim=512, depth=2):
        super().__init__()
        embed_dim, mlp_dim, depth = map(int, (embed_dim, mlp_dim, depth))
        if min(embed_dim, mlp_dim, depth) <= 0:
            raise ValueError("LatentFlowResidualClosure dimensions and depth must be positive")
        self.embed_dim = embed_dim
        layers = [nn.Linear(2 * embed_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, embed_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, delta_hat):
        if z_t.shape != delta_hat.shape or z_t.size(-1) != self.embed_dim:
            raise ValueError(
                f"LFR closure expects z_t/delta_hat=(..., {self.embed_dim}); "
                f"got {tuple(z_t.shape)} and {tuple(delta_hat.shape)}"
            )
        return self.net(torch.cat((z_t, delta_hat), dim=-1))


class IDMGapLatentFlowResidualClosure(nn.Module):
    """Latent-flow closure conditioned on an inverse-dynamics action gap.

    The gap ``g = a - I(z_t, P(z_t, a))`` is an online diagnostic of how
    completely the frozen predictor realizes its input action.  It is
    available for every CEM candidate at inference, while the MLP itself
    receives neither a goal nor a true future latent.
    """

    def __init__(self, embed_dim, action_dim, mlp_dim=512, depth=2):
        super().__init__()
        embed_dim, action_dim, mlp_dim, depth = map(
            int, (embed_dim, action_dim, mlp_dim, depth)
        )
        if min(embed_dim, action_dim, mlp_dim, depth) <= 0:
            raise ValueError("IDM-gap LFR dimensions and depth must be positive")
        self.embed_dim, self.action_dim = embed_dim, action_dim
        layers = [nn.Linear(2 * embed_dim + action_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, embed_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, delta_hat, action_gap):
        if z_t.shape != delta_hat.shape or z_t.shape[:-1] != action_gap.shape[:-1]:
            raise ValueError("IDM-gap LFR inputs must share leading dimensions")
        if z_t.size(-1) != self.embed_dim or action_gap.size(-1) != self.action_dim:
            raise ValueError(
                f"expected latent/action-gap dims {self.embed_dim}/{self.action_dim}, got "
                f"{z_t.size(-1)}/{action_gap.size(-1)}"
            )
        return self.net(torch.cat((z_t, delta_hat, action_gap), dim=-1))


class PlanActionCorrector(nn.Module):
    """Endpoint-conditioned residual IDM for an entire MPC plan.

    Unlike :class:`ActionCorrector`, this network has no one-transition
    assumption.  It receives a flattened macro-action plan ``U``, its start
    latent ``z_t``, and the endpoint rollout residual ``z_goal - P^H(z_t,U)``,
    then predicts a correction for all plan coordinates at once.
    """

    def __init__(self, plan_dim, latent_dim, mlp_dim=512, depth=3):
        super().__init__()
        plan_dim, latent_dim, mlp_dim, depth = map(
            int, (plan_dim, latent_dim, mlp_dim, depth)
        )
        if min(plan_dim, latent_dim, mlp_dim, depth) <= 0:
            raise ValueError("PlanActionCorrector dimensions and depth must be positive")
        layers = [nn.Linear(plan_dim + 2 * latent_dim, mlp_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(mlp_dim, mlp_dim), nn.SiLU()])
        layers.append(nn.Linear(mlp_dim, plan_dim))
        self.net = nn.Sequential(*layers)


        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, plan, z_t, endpoint_residual):
        if plan.ndim != 3 or z_t.shape[:-1] != plan.shape[:1] or endpoint_residual.shape != z_t.shape:
            raise ValueError("expected plan=(B,H,A), z_t/residual=(B,D)")
        return self.net(torch.cat((plan.flatten(1), z_t, endpoint_residual), dim=-1)).view_as(plan)


class ActionConsistencyIDM(nn.Module):
    """Inverse model used as an action-consistency discriminator.

    It consumes a state and a latent *transition*, ``(z_t, z_{t+1} - z_t)``,
    rather than two opaque endpoint tokens.  Its negative action MSE is the
    discriminator score used by the forward--inverse game.
    """

    def __init__(self, embed_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t, delta_z):
        return self.net(torch.cat([z_t, delta_z], dim=-1))


class CAIBlindPredictor(nn.Module):
    """State-only action estimator for the CAI behavior-policy baseline."""

    def __init__(self, embed_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t):
        return self.net(z_t)


class CAITransitionPredictor(InverseModel):
    """Transition action estimator used by Causal Action Identifiability."""

    pass


class BlindPredictor(nn.Module):
    """Action-free next-state predictor: z_t -> best action-free E[z_{t+1} | z_t].

    Used by IIPDCHead as the causal origin against which action-conditioned
    innovation is measured -- not a decoy adversary, not a counterfactual
    world. It defines "what would happen anyway" so the residual against it
    isolates the part of the transition actually explained by the action.
    """

    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z_t):
        """z_t: (B, D) or (B, T, D). Returns predicted z_{t+1}, same shape."""
        return self.net(z_t)


class IIPDCHead(nn.Module):
    """Inverse-Isometric PDC (II-PDC): merges PDC-style grounding with inverse
    dynamics into one isometric residual constraint, using only the logged
    action a_t -- no shuffled/counterfactual actions, no extra sampling.

    Given the action-conditioned prediction g(z_t, a_t) and a stop-gradient
    blind (action-free) reference b(z_t), define the action innovation:

        d_t = g(z_t, a_t) - sg[b(z_t)]

    and require it to equal an isometric (column-orthonormal, Q^T Q = I_m)
    linear transport of the logged action:

        L = || d_t - Q @ a_t ||^2

    Because Q^T Q = I_m, this single MSE decomposes exactly into:

        || d_t - Q a_t ||^2
            = || Q^T d_t - a_t ||^2          (inverse dynamics: a_t is
                                               recoverable from d_t)
            + || (I - Q Q^T) d_t ||^2        (no hidden shortcut: the part of
                                               d_t orthogonal to Q's column
                                               space must vanish)

    So it is not "PDC plus an inverse dynamics head" -- it IS an isometric
    inverse dynamics constraint. Unlike a free-form inverse MLP, Q fixes the
    norm and pairwise angles of the action-to-latent map, which blocks the
    infinitesimal-code collapse an unconstrained inverse head permits
    (d_t = eps*U*a_t with an inverse head that rescales by 1/eps -- action
    stays "recoverable" while the actual latent motion vanishes).

    Q is a genuinely learnable d x m matrix, obtained via a differentiable
    QR on every forward call (same technique as AFTPredictor's tangent
    frame): Q^T Q = I_m holds exactly at every step without any manual
    retraction/Cayley-map training-loop hook, and gradients flow through the
    QR back to the free parameter matrix.
    """

    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)
        self.M = nn.Parameter(torch.randn(self.embed_dim, self.action_dim))

    def forward(self, pred, blind_ref, action):
        """
        pred:      (..., D) - g(z_t, a_t), the action-conditioned prediction.
        blind_ref: (..., D) - b(z_t), NOT yet stop-gradiented by the caller's
                   responsibility -- this method detaches it internally.
        action:    (..., action_dim) - the logged action a_t.
        Returns the scalar II-PDC loss.
        """
        Q = _orthonormal_frame(self.M)
        innovation = pred - blind_ref.detach()
        action_transport = action.float() @ Q.T
        return F.mse_loss(innovation.float(), action_transport)


class ASPDCHead(nn.Module):
    """Action-Secant PDC (AS-PDC).

    Prior PDC variants supervise the temporal *pixel* delta: they push
    Q(z_{t+1}-z_t) toward R(x_{t+1}-x_t). But R*delta_x is only an
    observation-space secant, not the true world-state displacement -- for a
    nonlinear renderer x=g(s) it is the wrong target, so strengthening it
    (Gram, minimum-lift) drives the encoder toward projected pixels rather
    than world state. AS-PDC instead supervises a *counterfactual action*
    delta: under the same state z_t, two different actions must produce a
    latent secant that equals the action difference in one global linear
    coordinate frame.

        d_act = z_{t+1} - F(z_t, a_cf)          (true future vs. counterfactual)
        L = || Q d_act - (a_norm - a_cf_norm) ||^2 ,   Q Q^T = I_m

    where a_cf is another logged action drawn by permuting the batch (a valid,
    in-distribution action, not a synthetic one) and a_norm uses the existing
    baseline action normalization. One endpoint is the *real* next embedding
    z_{t+1} (kept differentiable -- NOT stop-gradiented -- so this constrains
    the encoder, not only the predictor); the other is the predictor's
    counterfactual F(z_t, a_cf).

    Identifiability: if this reaches zero on a controllable, persistently
    excited state space, then for any base state and actions a, b:
    Q[h(u+a)-h(u+b)] = a-b, and since Q is orthogonal, h(v)-h(w)=Q^T(v-w),
    giving h(s)=Q^T s + c -- LeJEPA/LeWorldModel's orthogonal linear
    identifiability, obtained from intervention geometry rather than an
    imposed Gaussian prior. A Gaussian world then maps to a Gaussian
    embedding as a consequence, not a constraint.

    Q is a single learnable m x d matrix (Q Q^T = I_m via differentiable QR
    on every call), persistent across steps -- it must NOT be resampled, since
    the loss aligns latent secants to a fixed action-coordinate target.
    """

    def __init__(self, embed_dim, action_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)


        self.M = nn.Parameter(torch.randn(self.embed_dim, self.action_dim))

    def forward(self, z_next, cf_pred, action, action_cf):
        """
        z_next:    (..., D) - real next embedding z_{t+1} (keep differentiable).
        cf_pred:   (..., D) - counterfactual prediction F(z_t, a_cf).
        action:    (..., action_dim) - normalized logged action a_t.
        action_cf: (..., action_dim) - normalized counterfactual action a_cf.
        Returns the scalar AS-PDC loss.
        """
        Qf = _orthonormal_frame(self.M)
        d_act = z_next - cf_pred
        latent_delta = d_act.float() @ Qf
        action_delta = (action - action_cf).float()
        return F.mse_loss(latent_delta, action_delta)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def _sample_probe(embed_dim: int, k: int, device, dtype) -> torch.Tensor:
    """Draw a fresh Haar-random row-orthonormal probe Q in R^{k x embed_dim}.

    Q should be a probe, not a representation: a persistent frozen Q leaves a
    permanent null-space (ker(Q)) the encoder can hide ungrounded information
    in, and a learned Q can co-adapt with the encoder to shortcut-align onto
    whatever subspace is easiest rather than the true full displacement. This
    avoids both by resampling Q every forward call under no_grad (so it never
    receives or blocks gradient), with E_Q[Q^T Q] = (k/embed_dim) I supplying
    an unbiased estimate of the full-space inner product once rescaled by
    sqrt(embed_dim/k): E_Q[(scale Q d_i)^T (scale Q d_j)] = d_i^T d_j exactly.

    When k == embed_dim, Q is a full change of basis (Q^T Q = I_d) and any
    orthonormal choice is equivalent to the identity for Gram/inner-product
    purposes, so this returns I_d directly instead of paying for a QR (and
    to sidestep the degenerate "reduced" mode when k == embed_dim).
    """
    k = int(k)
    if k >= embed_dim:
        return torch.eye(embed_dim, device=device, dtype=dtype)
    with torch.no_grad():
        M = torch.randn(embed_dim, k, device=device, dtype=torch.float32)
        Q_cols, _ = torch.linalg.qr(M, mode="reduced")
        Q = Q_cols.T.contiguous() * (embed_dim / k) ** 0.5
    return Q.to(dtype)


class PDCHead(torch.nn.Module):
    """Projected Delta Consistency (PDC) grounding loss.

    Requires latent displacement to preserve a fixed, non-learnable
    low-dimensional linear measurement of the realized pixel change:

        L_delta = || Q(z_{t+1} - z_t) - R(x_{t+1} - x_t) ||^2

    R (random sign projection, pixel space -> R^k) is frozen at init and
    never updated. Q (row-orthonormal, embed space -> R^k, Q Q^T = I_k)
    starts from an init-time draw and is then periodically resampled
    (every `resample_every` training steps, in-place, stop-gradient) rather
    than left permanently frozen or made learnable:

    - A *permanently frozen* Q leaves a fixed null-space (ker Q) the encoder
      can learn to park ungrounded information in for the entire run, since
      that blind spot never moves.
    - A *learned* Q can co-adapt with the encoder to shortcut-align onto
      whichever subspace is easiest to match, rather than the true full
      displacement.
    - Resampling *every step* (as GeoPDCHead does) doesn't work here: this
      loss is a per-coordinate MSE against a fixed target R*delta_x, and
      since E_Q[Q] = 0, resampling every call drives the cross term
      E_Q[(Q delta_z) . (R delta_x)] to ~0 in expectation, destroying the
      only part of the loss that actually supervises alignment (see
      `_sample_probe` docstring / GeoPDCHead, where the Gram/inner-product
      form makes per-step resampling valid instead).

    Periodic resampling is the middle ground: within a period, Q is fixed
    long enough for gradient descent to actually align delta_z to it: across
    periods, Q sweeps through different subspaces of the embedding, so no
    single null-space can hide ungrounded information for the whole run.

    Optional minimum-lift term (perp_weight > 0): plain PDC only constrains
    the k-dimensional projection Q*delta_z; any delta_z = Q^T R delta_x + n
    with n in ker(Q) satisfies it exactly, so the encoder can still hide
    arbitrary (nonlinear, non-Gaussian) information in that (d-k)-dimensional
    null space. Adding

        L_perp = || (I - Q^T Q) delta_z ||^2 / (d - k)

    penalizes exactly that leftover component (Q^T Q is the idempotent
    projector onto Q's row space, so I - Q^T Q projects onto its orthogonal
    complement). Together, L_delta + L_perp is minimized by the least-norm
    solution delta_z* = Q^T R delta_x -- the encoder is no longer free to
    add anything in ker(Q). When k == embed_dim, Q^T Q = I_d exactly (Q is a
    full change of basis) so L_perp is identically zero and this reduces to
    plain PDC, matching the k == d degenerate case.

    Only the encoder receives gradient through this loss (the predictor is
    not involved at all); Q is resampled under no_grad and never itself
    receives gradient.
    """

    def __init__(
        self,
        embed_dim,
        image_shape,
        k=64,
        seed=None,
        resample_every=None,
        perp_weight=0.0,
    ):
        super().__init__()
        c, h, w = image_shape
        self.embed_dim = int(embed_dim)
        self.k = int(k)
        self.perp_weight = float(perp_weight)



        self.resample_every = int(resample_every) if resample_every else None
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))

        signs = torch.randint(0, 2, (self.k, c * h * w), generator=generator).float()
        R = (signs * 2 - 1) / (self.k**0.5)
        self.register_buffer("R", R)

        M = torch.randn(embed_dim, self.k, generator=generator)
        Q_cols, _ = torch.linalg.qr(M, mode="reduced")
        self.register_buffer("Q", Q_cols.T.contiguous())
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def _resample_Q(self):
        with torch.no_grad():
            M = torch.randn(self.embed_dim, self.k, device=self.Q.device)
            Q_cols, _ = torch.linalg.qr(M, mode="reduced")
            self.Q.copy_(Q_cols.T.to(self.Q.dtype))

    def forward(self, z, x):
        """
        z: (B, T, D)          - encoder embeddings for consecutive frames.
        x: (B, T, C, H, W)    - resized (NOT normalized) pixels, same T as z.
        Returns scalar MSE between the projected latent displacement and the
        projected realized pixel displacement.
        """
        if self.training and self.resample_every is not None:
            if self._step.item() > 0 and self._step.item() % self.resample_every == 0:
                self._resample_Q()
            self._step += 1

        delta_z = (z[:, 1:] - z[:, :-1]).float()
        delta_x = x[:, 1:].float() - x[:, :-1].float()
        b, tm1 = delta_x.shape[:2]
        delta_x_flat = delta_x.reshape(b, tm1, -1)

        r = torch.einsum("btf,kf->btk", delta_x_flat, self.R)
        r_hat = torch.einsum("btd,kd->btk", delta_z, self.Q)
        loss = F.mse_loss(r_hat, r)

        if self.perp_weight and self.embed_dim > self.k:




            dz_parallel = torch.einsum("btk,kd->btd", r_hat, self.Q)
            dz_perp = delta_z - dz_parallel
            loss_perp = dz_perp.pow(2).sum(dim=-1).mean() / (self.embed_dim - self.k)
            loss = loss + self.perp_weight * loss_perp

        return loss

    def forward_jpdc(self, z_mid_pred, z_next, z_other_pred):
        """Jensen-PDC: action-midpoint consistency (second-order curvature term).

        For a controlled-additive world s_{t+1}^a = f(s_t, xi_t) + B(s_t) a,
        the midpoint action a_m = (a+a')/2 gives s_{t+1}^{a_m} = (s_{t+1}^a +
        s_{t+1}^{a'})/2 identically. A representation h that preserves this
        midpoint relation for every action pair, on a connected action set,
        must be affine (Jensen's functional equation) -- this rules out
        nonlinear bending of the encoder along the action direction, without
        pinning latent displacement to any specific coordinate frame or scale
        (unlike matching Qa directly).

        Deliberately projected through this head's own Q (the same frozen
        low-rank projector plain PDC already uses) rather than the full
        embed_dim: this constrains the same k-dim subspace plain PDC grounds,
        instead of imposing a fresh constraint over the whole latent space.

        z_mid_pred:   (..., D) - F(z_t, (a+a_cf)/2), kept differentiable.
        z_next:       (..., D) - real next embedding z_{t+1}, kept
                      differentiable (NOT stop-gradiented -- otherwise this
                      only trains the predictor's interpolation, not the
                      encoder).
        z_other_pred: (..., D) - F(z_t, a_cf), kept differentiable.
        """
        midpoint_gap = (z_mid_pred - 0.5 * (z_next + z_other_pred)).float()
        proj = torch.einsum("...d,kd->...k", midpoint_gap, self.Q)
        return proj.pow(2).mean()


class GeoPDCHead(torch.nn.Module):
    """Geometric Projected Delta Consistency (geo-PDC).

    Vanilla PDC forces `Q * delta_z == R * delta_x` coordinate-for-coordinate,
    which pins latent displacement to R's arbitrary frozen coordinate frame.
    This instead only requires the two projected residual sets to share the
    same pairwise Gram (inner-product) structure over a trajectory window:

        (Q Delta_z)(Q Delta_z)^T  ~=  (R Delta_x)(R Delta_x)^T

    Two residual sets with equal Gram matrices differ by at most a global
    orthogonal transform O, i.e. Q Delta_z = O^T R Delta_x, so this preserves
    residual norms and pairwise angles without pinning latent displacement to
    R's specific axes -- the reverse-direction analogue of LeJEPA's z = O s
    identifiability, derived from transition geometry instead of an imposed
    Gaussian prior. If the encoder collapses (Delta_z = 0), the Gram matrix of
    the latent side is identically zero while the pixel side's Gram matrix is
    not (whenever pixels actually change), so this keeps the same
    anti-collapse guarantee as vanilla PDC.

    Computed at multiple temporal scales (e.g. 1, 2, 4 steps) so both local
    edge lengths and longer-range trajectory shape are constrained -- a
    shape-preserving-but-folded encoding could satisfy a single-step
    constraint but not a multi-scale one.
    """

    def __init__(self, embed_dim, image_shape, k=64, scales=(1, 2, 4), seed=None):
        super().__init__()
        c, h, w = image_shape
        self.k = int(k)
        self.scales = tuple(int(m) for m in scales)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))

        signs = torch.randint(0, 2, (self.k, c * h * w), generator=generator).float()
        R = (signs * 2 - 1) / (self.k**0.5)
        self.register_buffer("R", R)

        M = torch.randn(embed_dim, self.k, generator=generator)
        Q_cols, _ = torch.linalg.qr(M, mode="reduced")
        self.register_buffer("Q", Q_cols.T.contiguous())

    def forward(self, z, x):
        """
        z: (B, T, D)       - encoder embeddings for consecutive frames.
        x: (B, T, C, H, W) - resized (NOT normalized) pixels, same T as z.
        Returns scalar mean (over used scales) of squared Frobenius Gram
        mismatch between projected latent and pixel residual sets. Scales
        with m >= T are skipped (not enough frames in this batch).
        """
        z = z.float()
        x = x.float()
        b, t = z.shape[:2]
        x_flat = x.reshape(b, t, -1)

        total = z.new_zeros(())
        n_used = 0
        for m in self.scales:
            if m >= t:
                continue
            delta_z = z[:, m:] - z[:, :-m]
            delta_x = x_flat[:, m:] - x_flat[:, :-m]

            u = torch.einsum("btd,kd->btk", delta_z, self.Q)
            v = torch.einsum("btf,kf->btk", delta_x, self.R)

            gram_u = torch.einsum("btk,bsk->bts", u, u)
            gram_v = torch.einsum("btk,bsk->bts", v, v)

            total = total + (gram_u - gram_v).pow(2).mean()
            n_used += 1

        if n_used == 0:
            raise ValueError(
                f"GeoPDCHead: no usable scale in {self.scales} for sequence "
                f"length T={t} (need at least one scale m < T)."
            )
        return total / n_used


class DeltaLiftHead(torch.nn.Module):
    """Grounded DeltaLift: an anchored-identity anti-collapse loss.

    Plain norm barriers on latent displacement (e.g. 1/(|delta_z|^2+eps) or
    relu(tau-|delta_z|)^2) are rotation-invariant functions of |delta_z|
    alone, so their gradient at the exact collapse point delta_z=0 is
    identically zero (grad of h(|d|^2) at d=0 is 2 h'(0) d = 0): they can make
    the collapsed state high-loss, but they never tell the encoder *which
    direction* to move to escape it, and a barrier that also keeps pushing
    outside the danger zone (like 1/|d|^2) fights the forward loss forever
    instead of getting out of its way once displacement is already healthy.

    DeltaLift instead defines an explicit anchored-identity map on the
    projected latent displacement d = Q(z_{t+1}-z_t) (Q: fixed row-orthonormal
    probe, embed space -> R^k):

        g_tau(d)   = relu(1 - |d|^2 / tau^2)                   in [0, 1]
        M_tau(d,u) = (1 - g_tau) * d  +  g_tau * tau * u
        L          = || d - M_tau(d, u) ||^2 / tau^2
                   = g_tau^2 * || d - tau*u ||^2 / tau^2

    u = R(x_{t+1}-x_t) / ||R(x_{t+1}-x_t)|| is a fixed, non-learnable escape
    direction taken from the *real* sensory change (R: frozen random sign
    projection, pixel space -> R^k, reused from PDCHead's construction) --
    not an arbitrary fixed vector, since the direction the encoder should
    expand into is whatever direction the actual observation actually moved.

    Outside the near-zero ball (|d| >= tau): g_tau = 0, M_tau(d,u) = d
    identically, so L = 0 with zero gradient -- normal transitions are
    completely untouched, unlike a barrier that keeps a residual pull at all
    scales.

    Inside the ball, and in particular exactly at collapse (d=0, so
    g_tau=1, M_tau(0,u)=tau*u):

        L(0) = 1        (nonzero, unlike a rotation-invariant penalty)
        grad_d L |_{d=0} = -(2/tau) u  =/= 0

    so the exact constant-embedding fixed point is no longer a stationary
    point of this loss -- it has an explicit, grounded escape direction.

    Transitions where the scene genuinely does not change (||R delta_x|| ~ 0,
    e.g. a held-still action) are excluded from the loss entirely (rather
    than being forced toward an arbitrary escape direction u that is only
    noise in that case): those samples do not contribute to the mean.

    Q and R are both fixed once at init (no resampling, no learning): the
    escape direction must stay tied to one consistent coordinate frame for
    the anchoring identity above to hold at every step.
    """

    def __init__(
        self,
        embed_dim,
        image_shape,
        k=64,
        tau=1.0,
        seed=None,
        pixel_eps=1e-6,
    ):
        super().__init__()
        c, h, w = image_shape
        self.embed_dim = int(embed_dim)
        self.k = int(k)
        self.tau = float(tau)
        self.pixel_eps = float(pixel_eps)

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))

        signs = torch.randint(0, 2, (self.k, c * h * w), generator=generator).float()
        R = (signs * 2 - 1) / (self.k**0.5)
        self.register_buffer("R", R)

        M = torch.randn(embed_dim, self.k, generator=generator)
        Q_cols, _ = torch.linalg.qr(M, mode="reduced")
        self.register_buffer("Q", Q_cols.T.contiguous())

    def forward(self, z, x):
        """
        z: (B, T, D)       - encoder embeddings for consecutive frames.
        x: (B, T, C, H, W) - resized (NOT normalized) pixels, same T as z.
        Returns the scalar DeltaLift loss, averaged only over transitions
        with genuine (non-degenerate) sensory change.
        """
        delta_z = (z[:, 1:] - z[:, :-1]).float()
        delta_x = x[:, 1:].float() - x[:, :-1].float()
        b, tm1 = delta_x.shape[:2]
        delta_x_flat = delta_x.reshape(b, tm1, -1)

        d = torch.einsum("btd,kd->btk", delta_z, self.Q)
        s = torch.einsum("btf,kf->btk", delta_x_flat, self.R)
        s_norm = s.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            u = s / s_norm.clamp_min(self.pixel_eps)
            moving = (s_norm.squeeze(-1) > self.pixel_eps).float()

        r2 = d.square().sum(dim=-1, keepdim=True)
        gate = F.relu(1.0 - r2 / self.tau**2)
        mapped_d = (1.0 - gate) * d + gate * self.tau * u

        loss_per = (d - mapped_d).square().sum(dim=-1) / self.tau**2
        return (moving * loss_per).sum() / moving.sum().clamp_min(1.0)
