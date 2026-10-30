"""Convert unified network outputs to runtime detections."""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import nms

from edgefall_st.runtime.types import Detection


def postprocess_predictions(
    outputs: dict[str, Tensor],
    *,
    person_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> list[list[Detection]]:
    boxes = outputs["boxes"]
    person_scores = outputs["objectness_logits"].sigmoid()
    stage_probabilities = outputs["stage_logits"].softmax(dim=-1)
    onset_scores = outputs["onset_logits"].sigmoid()
    embeddings = outputs["track_embeddings"]

    batch_detections: list[list[Detection]] = []
    for batch_index in range(boxes.shape[0]):
        keep_mask = person_scores[batch_index] >= person_threshold
        candidate_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        if candidate_indices.numel() == 0:
            batch_detections.append([])
            continue
        candidate_boxes = boxes[batch_index, candidate_indices]
        candidate_scores = person_scores[batch_index, candidate_indices]
        kept = nms(
            candidate_boxes,
            candidate_scores,
            nms_iou_threshold,
        )[:max_detections]
        selected = candidate_indices[kept]
        detections = [
            Detection(
                bbox=tuple(
                    float(value)
                    for value in boxes[batch_index, index].detach().cpu().tolist()
                ),
                person_score=float(
                    person_scores[batch_index, index].detach().cpu()
                ),
                stage_probabilities=tuple(
                    float(value)
                    for value in stage_probabilities[
                        batch_index, index
                    ].detach().cpu().tolist()
                ),
                onset_score=float(
                    onset_scores[batch_index, index].detach().cpu()
                ),
                embedding=tuple(
                    float(value)
                    for value in embeddings[
                        batch_index, index
                    ].detach().cpu().tolist()
                ),
            )
            for index in selected.tolist()
        ]
        batch_detections.append(detections)
    return batch_detections
