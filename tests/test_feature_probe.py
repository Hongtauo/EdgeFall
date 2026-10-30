from __future__ import annotations

import torch

from edgefall_st.evaluation.feature_probe import (
    _masked_mean_and_delta_batch,
    summarize_batch,
)


def test_masked_mean_and_delta_uses_first_and_last_valid_values() -> None:
    values = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]],
            [[3.0, 5.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )
    mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    summary = _masked_mean_and_delta_batch(values, mask)

    torch.testing.assert_close(
        summary,
        torch.tensor(
            [
                [3.0, 6.0, 2.0, 4.0],
                [3.0, 5.0, 0.0, 0.0],
            ]
        ),
    )


def test_summarize_batch_supports_single_and_multiple_people() -> None:
    batch = {
        "global_features": torch.ones(2, 2, 3),
        "roi_features": torch.ones(2, 2, 4, 2, 2),
        "actor_boxes": torch.ones(2, 3, 4),
        "valid_mask": torch.ones(2, 3),
        "actor_valid_mask": torch.ones(2, 3),
    }
    single = summarize_batch(batch, visual_slots=(0, 2))
    assert single["geometry"].shape == (2, 23)
    assert single["global"].shape == (2, 6)
    assert single["roi"].shape == (2, 8)
    assert single["visual"].shape == (2, 14)

    multi_batch = dict(batch)
    multi_batch["roi_features"] = batch["roi_features"].unsqueeze(2).repeat(
        1, 1, 2, 1, 1, 1
    )
    multi_batch["actor_boxes"] = batch["actor_boxes"].unsqueeze(2).repeat(
        1, 1, 2, 1
    )
    multi_batch["actor_valid_mask"] = batch["actor_valid_mask"].unsqueeze(
        2
    ).repeat(1, 1, 2)
    multiple = summarize_batch(multi_batch, visual_slots=(0, 2))
    for name in single:
        torch.testing.assert_close(multiple[name], single[name])
