"""Siamese similarity model on top of a pretrained code encoder.

Both files go through the *same* encoder (shared weights). Token states are mean-pooled
over the attention mask and projected to ``projection_dim``. Two heads are available:

* ``mlp`` (default): an MLP over ``[u, v, |u - v|, u * v]``. To keep the score symmetric the
  logit is averaged over both argument orders.
* ``cosine``: ``scale * cos(u, v) + bias`` with learned scale and bias.

The model outputs a logit; ``sigmoid(logit)`` is the neural similarity score.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from codeplag.core.config import NeuralConfig
from codeplag.core.logging import get_logger

logger = get_logger(__name__)


def mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    """Average token states where ``mask`` is 1.

    Args:
        hidden: ``[batch, seq, dim]`` token states.
        mask: ``[batch, seq]`` attention mask.

    Returns:
        ``[batch, dim]`` pooled embeddings.
    """
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    return summed / weights.sum(dim=1).clamp(min=1e-6)


class SiameseCodeModel(nn.Module):
    """Shared encoder + pooling + projection + similarity head.

    Args:
        encoder: A module returning an object with ``last_hidden_state`` (Hugging Face style)
            when called with ``input_ids`` and ``attention_mask``.
        hidden_size: Encoder output dimension.
        projection_dim: Size of the projected embedding.
        head: ``mlp`` or ``cosine``.
    """

    def __init__(
        self, encoder: nn.Module, hidden_size: int, projection_dim: int = 256, head: str = "mlp"
    ) -> None:
        super().__init__()
        if head not in ("mlp", "cosine"):
            raise ValueError(f"Unknown head {head!r}")
        self.encoder = encoder
        self.head_type = head
        self.projection = nn.Sequential(nn.Linear(hidden_size, projection_dim), nn.Tanh())
        if head == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(4 * projection_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim, 1),
            )
        else:
            self.scale = nn.Parameter(torch.tensor(10.0))
            self.bias = nn.Parameter(torch.tensor(-5.0))

    def embed(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Encode a batch of token sequences into projected embeddings."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        return self.projection(mean_pool(hidden, attention_mask))

    def pair_logits(self, u: Tensor, v: Tensor) -> Tensor:
        """Similarity logits for paired embeddings ``u[i]``, ``v[i]``."""
        if self.head_type == "cosine":
            return self.scale * F.cosine_similarity(u, v, dim=-1) + self.bias
        diff, prod = (u - v).abs(), u * v
        forward = self.classifier(torch.cat([u, v, diff, prod], dim=-1))
        backward = self.classifier(torch.cat([v, u, diff, prod], dim=-1))
        return ((forward + backward) / 2).squeeze(-1)

    def forward(self, ids_a: Tensor, mask_a: Tensor, ids_b: Tensor, mask_b: Tensor) -> Tensor:
        """Return similarity logits for a batch of pairs."""
        return self.pair_logits(self.embed(ids_a, mask_a), self.embed(ids_b, mask_b))


def freeze_lower_layers(encoder: nn.Module, n_layers: int) -> int:
    """Freeze the embeddings and the first ``n_layers`` transformer layers.

    Works with BERT/RoBERTa-style encoders (``embeddings`` and ``encoder.layer``). Unknown
    architectures are left untouched.

    Args:
        encoder: Hugging Face encoder.
        n_layers: Number of lower layers to freeze (0 disables freezing).

    Returns:
        Number of frozen parameters.
    """
    if n_layers <= 0:
        return 0
    modules: list[nn.Module] = []
    embeddings = getattr(encoder, "embeddings", None)
    if isinstance(embeddings, nn.Module):
        modules.append(embeddings)
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if isinstance(layers, nn.ModuleList):
        modules.extend(layers[:n_layers])
    frozen = 0
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
            frozen += param.numel()
    logger.info("Froze %d encoder parameters (%d layers + embeddings)", frozen, n_layers)
    return frozen


def load_encoder(model_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Load a Hugging Face encoder.

    Args:
        model_name: Hub id or local path.
        pretrained: Load pretrained weights; otherwise only the architecture (used when a
            checkpoint will overwrite the weights anyway).

    Returns:
        The encoder and its hidden size.
    """
    from transformers import AutoConfig, AutoModel

    if pretrained:
        encoder = AutoModel.from_pretrained(model_name)
    else:
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
    return encoder, int(encoder.config.hidden_size)


def build_model(
    config: NeuralConfig,
    encoder: nn.Module | None = None,
    hidden_size: int | None = None,
    pretrained: bool = True,
) -> SiameseCodeModel:
    """Build a :class:`SiameseCodeModel` from settings.

    Args:
        config: Neural settings.
        encoder: Pre-built encoder (tests inject a small mock here).
        hidden_size: Hidden size of ``encoder`` (required when ``encoder`` is given).
        pretrained: Passed to :func:`load_encoder` when ``encoder`` is not given.

    Returns:
        The model, with lower layers frozen per ``config.freeze_layers``.
    """
    if encoder is None:
        encoder, hidden_size = load_encoder(config.model_name, pretrained=pretrained)
    if hidden_size is None:
        raise ValueError("hidden_size is required when passing an encoder")
    freeze_lower_layers(encoder, config.freeze_layers)
    return SiameseCodeModel(encoder, hidden_size, config.projection_dim, config.head)


def resolve_device(name: str) -> torch.device:
    """Map a device name to :class:`torch.device`, falling back to CPU if CUDA is missing."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def checkpoint_payload(
    model: SiameseCodeModel, config: NeuralConfig, **extra: Any
) -> dict[str, Any]:
    """Build the dictionary saved to a checkpoint file."""
    return {"state_dict": model.state_dict(), "neural_config": config.model_dump(), **extra}
