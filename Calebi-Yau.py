"""
calabibrain.py

A toy architecture inspired by Richard's idea of using a Calabi–Yau-like
compactified space to model multi-scale brain behaviour.

Concept:
- Macro state: coarse, observable behaviour (e.g. neurons, columns, graph nodes)
- Micro state: hidden compactified dimensions (Calabi–Yau-ish), one fibre per macro unit
- Micro dynamics influence macro via an "effective field" but are never observed directly

This is NOT physics-grounded, just a fun geometrically-inspired NN block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroDynamics(nn.Module):
    """
    Local dynamics in the compactified micro-space.
    Think: microtubules / subcellular chaos / internal oscillations,
    all squashed into a small latent fibre per macro unit.
    """

    def __init__(self, micro_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(micro_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, micro_dim),
        )

    def forward(self, micro_state):
        """
        micro_state: [batch, N_macro, D_micro]
        """
        # Apply the same dynamics independently to each macro unit's fibre
        # Flatten batch + macro, run MLP, then reshape
        b, n, d = micro_state.shape
        x = micro_state.reshape(b * n, d)
        x = self.net(x)
        x = x.reshape(b, n, d)
        return x


class MicroToMacroCoupling(nn.Module):
    """
    Computes an effective macro field from the hidden micro-space.
    This is the 'integrating out' of micro degrees of freedom.

    Here we do:
    - project micro -> influence space
    - nonlinearity
    - aggregate back to macro via a learned gate
    """

    def __init__(self, macro_dim: int, micro_dim: int, hidden_dim: int):
        super().__init__()
        self.micro_proj = nn.Linear(micro_dim, hidden_dim)
        self.macro_gate = nn.Linear(macro_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, macro_dim)

    def forward(self, macro_state, micro_state):
        """
        macro_state: [batch, N_macro, D_macro]
        micro_state: [batch, N_macro, D_micro]
        returns: effective_field [batch, N_macro, D_macro]
        """
        micro_feat = self.micro_proj(micro_state)      # [b, n, h]
        gate = torch.sigmoid(self.macro_gate(macro_state))  # [b, n, h]
        combined = F.gelu(micro_feat * gate)           # gated micro signal
        effective = self.out_proj(combined)            # [b, n, D_macro]
        return effective


class MacroInteraction(nn.Module):
    """
    Optional macro–macro interaction (small-world / graph-ish behaviour).
    Here: a simple self-attention over macro units.
    """

    def __init__(self, macro_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=macro_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, macro_state):
        """
        macro_state: [batch, N_macro, D_macro]
        """
        # Self-attention over macro units
        out, _ = self.attn(macro_state, macro_state, macro_state)
        return out


class CalabiYauBlock(nn.Module):
    """
    One Calabi–Yau brain block:
    - update compactified micro-space
    - compute effective field from micro -> macro
    - update macro, plus macro–macro interaction
    """

    def __init__(
        self,
        macro_dim: int,
        micro_dim: int,
        micro_hidden: int,
        coupling_hidden: int,
        macro_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.micro_dyn = MicroDynamics(micro_dim, micro_hidden)
        self.coupling = MicroToMacroCoupling(macro_dim, micro_dim, coupling_hidden)
        self.macro_interaction = MacroInteraction(macro_dim, num_heads=macro_heads)

        self.macro_norm1 = nn.LayerNorm(macro_dim)
        self.macro_norm2 = nn.LayerNorm(macro_dim)
        self.micro_norm = nn.LayerNorm(micro_dim)

        self.macro_ff = nn.Sequential(
            nn.Linear(macro_dim, 4 * macro_dim),
            nn.GELU(),
            nn.Linear(4 * macro_dim, macro_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, macro_state, micro_state):
        """
        macro_state: [batch, N_macro, D_macro]
        micro_state: [batch, N_macro, D_micro]

        returns:
            new_macro_state, new_micro_state
        """

        # 1. Update micro dynamics (compactified space evolution)
        residual_micro = micro_state
        micro_state = self.micro_dyn(micro_state)
        micro_state = self.micro_norm(micro_state + residual_micro)

        # 2. Micro -> Macro effective field
        effective_field = self.coupling(macro_state, micro_state)

        # 3. Macro update: macro–macro interaction + effective field
        residual_macro = macro_state
        macro_inter = self.macro_interaction(macro_state)
        macro_state = macro_state + self.dropout(macro_inter + effective_field)
        macro_state = self.macro_norm1(macro_state)

        # 4. Macro feedforward
        residual_macro2 = macro_state
        macro_ff = self.macro_ff(macro_state)
        macro_state = residual_macro2 + self.dropout(macro_ff)
        macro_state = self.macro_norm2(macro_state)

        return macro_state, micro_state


class CalabiYauBrain(nn.Module):
    """
    Full model: stack several Calabi–Yau blocks.

    You provide:
    - N_macro: number of macro units (e.g. nodes, neurons, patches)
    - macro_dim: size of each macro embedding
    - micro_dim: size of compactified micro-space per macro unit

    This class only defines the generic architecture.
    You still need a task-specific input encoder and output head.
    """

    def __init__(
        self,
        macro_dim: int = 64,
        micro_dim: int = 16,
        depth: int = 4,
        micro_hidden: int = 32,
        coupling_hidden: int = 64,
        macro_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.macro_dim = macro_dim
        self.micro_dim = micro_dim

        self.blocks = nn.ModuleList(
            [
                CalabiYauBlock(
                    macro_dim=macro_dim,
                    micro_dim=micro_dim,
                    micro_hidden=micro_hidden,
                    coupling_hidden=coupling_hidden,
                    macro_heads=macro_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, macro_state, micro_state=None):
        """
        macro_state: [batch, N_macro, D_macro]
        micro_state (optional): [batch, N_macro, D_micro]

        If micro_state is None, it is initialised as zeros.
        """

        if micro_state is None:
            b, n, _ = macro_state.shape
            device = macro_state.device
            micro_state = torch.zeros(b, n, self.micro_dim, device=device)

        for block in self.blocks:
            macro_state, micro_state = block(macro_state, micro_state)

        return macro_state, micro_state


if __name__ == "__main__":
    # Tiny sanity check
    batch_size = 2
    N_macro = 8
    macro_dim = 32
    micro_dim = 12

    model = CalabiYauBrain(
        macro_dim=macro_dim,
        micro_dim=micro_dim,
        depth=3,
    )

    x_macro = torch.randn(batch_size, N_macro, macro_dim)
    macro_out, micro_out = model(x_macro)

    print("macro_out:", macro_out.shape)
    print("micro_out:", micro_out.shape)