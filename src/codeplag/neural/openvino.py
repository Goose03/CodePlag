"""Optional FP32 OpenVINO runtime for the UniXcoder embedding encoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from codeplag.core.logging import get_logger
from codeplag.neural.model import SiameseCodeModel

logger = get_logger(__name__)


class _EmbeddingEncoder(nn.Module):
    def __init__(self, model: SiameseCodeModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        return self.model.embed(ids, mask)


@torch.no_grad()
def compile_encoder(
    model: SiameseCodeModel, example: tuple[Tensor, Tensor], device: str
) -> Callable[[Tensor, Tensor], Tensor]:
    """Compile the embedding network for optional FP32 inference on CPU or Intel GPU.

    Args:
        model: Evaluated siamese model with weights on CPU.
        example: Two tokenized sources with different lengths in one padded batch.
        device: OpenVINO device name.

    Returns:
        A callable that accepts CPU input IDs and masks and returns a CPU tensor.
    """
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("Install codeplag[openvino] to use the OpenVINO encoder.") from exc

    core = ov.Core()
    network = ov.convert_model(_EmbeddingEncoder(model).eval(), example_input=example)
    config = {"INFERENCE_PRECISION_HINT": "f32"}
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = str(torch.get_num_threads())
    encoder = core.compile_model(network, device, config)
    logger.info("OpenVINO FP32 encoder on %s", core.get_property(device, "FULL_DEVICE_NAME"))

    def embed(ids: Tensor, mask: Tensor) -> Tensor:
        output = encoder([ids.numpy(), mask.numpy()])[0]
        return torch.from_numpy(output.copy())

    return embed
