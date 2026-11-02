import math

import pytest
import torch

from temporal_recsys.models.two_tower import in_batch_loss

IDENTITY = torch.eye(2)
ROWS = torch.tensor([0, 1])


def test_loss_with_no_popularity_correction() -> None:
    loss = in_batch_loss(IDENTITY, IDENTITY, ROWS, torch.zeros(2), temperature=1.0)

    # Each user scores 1 for their film and 0 for the other.
    assert loss.item() == pytest.approx(-math.log(math.e / (math.e + 1)), rel=1e-5)


def test_temperature_divides_the_scores() -> None:
    loss = in_batch_loss(IDENTITY, IDENTITY, ROWS, torch.zeros(2), temperature=0.5)

    assert loss.item() == pytest.approx(-math.log(math.e**2 / (math.e**2 + 1)), rel=1e-5)


def test_popularity_is_subtracted_from_each_films_column() -> None:
    log_popularity = torch.tensor([math.log(2.0), 0.0])

    loss = in_batch_loss(IDENTITY, IDENTITY, ROWS, log_popularity, temperature=1.0)

    # Dividing film 0's score by 2 gives each user these two chances of being right.
    first_user = (math.e / 2) / (math.e / 2 + 1)
    second_user = math.e / (math.e + 0.5)
    expected = -(math.log(first_user) + math.log(second_user)) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_a_film_repeated_in_the_batch_is_not_a_wrong_answer() -> None:
    same_film = torch.tensor([3, 3])

    loss = in_batch_loss(IDENTITY, IDENTITY, same_film, torch.zeros(4), temperature=1.0)

    assert loss.item() == pytest.approx(0.0)
