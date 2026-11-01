"""A two-tower model: one network describes the user and another describes the film.

The user tower averages the films a user rated recently, and has no per-user ID. The
film tower combines a film's ID with its release year and title words, so `use_content`
can turn those off to test whether they help. Both towers end in the same kind of
vector as ALS, so the trained model is exported as a `FactorModel` and scored the same way.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from temporal_recsys.data.arrays import RatingArrays
from temporal_recsys.data.films import FilmFeatures
from temporal_recsys.data.histories import (
    PADDING,
    UserHistories,
    build_histories,
    latest_inputs,
    training_inputs,
)
from temporal_recsys.models.factors import FactorModel

LOGGER = logging.getLogger(__name__)

EXPORT_BATCH_SIZE = 8_192


@dataclass(frozen=True)
class TwoTowerConfig:
    dimensions: int = 64
    max_history: int = 50
    batch_size: int = 4_096
    epochs: int = 3
    learning_rate: float = 0.001
    temperature: float = 0.05
    use_content: bool = True
    seed: int = 0


def in_batch_loss(
    user_vectors: Tensor,
    film_vectors: Tensor,
    film_rows: Tensor,
    log_popularity: Tensor,
    temperature: float,
) -> Tensor:
    """Ask each user to pick their own film out of all the films in the batch.

    Popular films land in batches more often, so they are punished more often as wrong
    answers. Subtracting each film's log popularity removes that bias (Yi et al., 2019).
    A film that appears twice in a batch is not counted as a wrong answer against itself.
    """
    logits = user_vectors @ film_vectors.T / temperature - log_popularity[film_rows]
    is_duplicate = film_rows[:, None] == film_rows[None, :]
    is_duplicate.fill_diagonal_(False)
    logits = logits.masked_fill(is_duplicate, -torch.inf)
    return F.cross_entropy(logits, torch.arange(len(film_rows), device=logits.device))


class TwoTowerNet(nn.Module):
    id_inputs: Tensor
    year_buckets: Tensor
    title_words: Tensor

    def __init__(
        self, features: FilmFeatures, is_in_training: NDArray[np.bool_], config: TwoTowerConfig
    ) -> None:
        super().__init__()
        n_films, size = len(is_in_training), config.dimensions
        self.use_content = config.use_content
        self.history_films = nn.Embedding(n_films + 1, size, padding_idx=PADDING)
        self.user_layers = _layers(size)
        self.film_ids = nn.Embedding(n_films + 1, size, padding_idx=PADDING)
        self.years = nn.Embedding(features.n_year_buckets, size, padding_idx=PADDING)
        self.words = nn.Embedding(features.n_words, size, padding_idx=PADDING)
        self.film_layers = _layers(size)

        # A film unseen in training has an ID vector that was never trained, so that
        # vector stays out, and only its title and year can describe it.
        id_inputs = np.where(is_in_training, np.arange(1, n_films + 1), PADDING)
        self.register_buffer("id_inputs", torch.from_numpy(id_inputs))
        self.register_buffer("year_buckets", torch.from_numpy(features.year_buckets))
        self.register_buffer("title_words", torch.from_numpy(features.title_words))

    def encode_users(self, histories: Tensor) -> Tensor:
        pooled = _mean_of_present(self.history_films(histories), histories)
        return F.normalize(self.user_layers(pooled), dim=1)

    def encode_films(self, film_rows: Tensor) -> Tensor:
        vectors = self.film_ids(self.id_inputs[film_rows])
        if self.use_content:
            words = self.title_words[film_rows]
            vectors = vectors + self.years(self.year_buckets[film_rows])
            vectors = vectors + _mean_of_present(self.words(words), words)
        return F.normalize(self.film_layers(vectors), dim=1)


class TwoTowerModel:
    def __init__(
        self,
        config: TwoTowerConfig,
        features: FilmFeatures,
        device: str = "cpu",
        checkpoint_path: Path | None = None,
    ) -> None:
        self.config = config
        self.features = features
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.training_seconds = 0.0
        self._factors: FactorModel | None = None

    def fit(self, train: RatingArrays, n_films: int) -> None:
        config = self.config
        torch.manual_seed(config.seed)
        histories = build_histories(train)
        counts = np.bincount(train.film_ids - 1, minlength=n_films)
        is_in_training = counts > 0
        log_popularity = np.log(np.maximum(counts, 1) / len(train))

        net = TwoTowerNet(self.features, is_in_training, config).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
        first_epoch = self._resume(net, optimizer)

        log_tensor = torch.from_numpy(log_popularity).float().to(self.device)
        for epoch in range(first_epoch, config.epochs):
            start = time.perf_counter()
            loss = self._train_epoch(net, optimizer, histories, log_tensor, epoch)
            self.training_seconds += time.perf_counter() - start
            self._save(net, optimizer, epoch + 1)
            LOGGER.info(
                "epoch %d: loss %.4f, %.1fs in total", epoch + 1, loss, self.training_seconds
            )
        self._factors = self._export(net, histories, is_in_training)

    def score_users(self, user_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        """Score every film. Users absent from training score zero for all films."""
        if self._factors is None:
            raise RuntimeError("Call fit before scoring")
        return self._factors.score_users(user_ids)

    @property
    def factors(self) -> FactorModel:
        if self._factors is None:
            raise RuntimeError("Call fit before reading factors")
        return self._factors

    def _train_epoch(
        self,
        net: TwoTowerNet,
        optimizer: torch.optim.Optimizer,
        histories: UserHistories,
        log_popularity: Tensor,
        epoch: int,
    ) -> float:
        config = self.config
        # Seeding from the epoch lets a restart continue the same sequence of batches.
        rng = np.random.default_rng([config.seed, epoch])
        rows = rng.permutation(histories.training_rows())
        if len(rows) < config.batch_size:
            raise ValueError(f"{len(rows)} training rows cannot fill one batch")

        net.train()
        losses: list[Tensor] = []
        for start in range(0, len(rows) - config.batch_size + 1, config.batch_size):
            batch = rows[start : start + config.batch_size]
            inputs = training_inputs(histories, batch, config.max_history)
            film_rows = torch.from_numpy(histories.film_ids[batch].astype(np.int64) - 1)
            film_rows = film_rows.to(self.device)

            user_vectors = net.encode_users(torch.from_numpy(inputs).to(self.device))
            film_vectors = net.encode_films(film_rows)
            loss = in_batch_loss(
                user_vectors, film_vectors, film_rows, log_popularity, config.temperature
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(loss.detach())
        return float(torch.stack(losses).mean())

    def _export(
        self, net: TwoTowerNet, histories: UserHistories, is_in_training: NDArray[np.bool_]
    ) -> FactorModel:
        net.eval()
        inputs = latest_inputs(histories, self.config.max_history)
        with torch.no_grad():
            users = [
                net.encode_users(torch.from_numpy(batch).to(self.device)).cpu().numpy()
                for batch in np.array_split(inputs, _batch_count(len(inputs)))
            ]
            all_rows = torch.arange(len(is_in_training), device=self.device)
            films = net.encode_films(all_rows).cpu().numpy().astype(np.float64)
        if not self.config.use_content:
            films[~is_in_training] = 0.0
        return FactorModel(histories.user_ids, np.concatenate(users).astype(np.float64), films)

    def _resume(self, net: TwoTowerNet, optimizer: torch.optim.Optimizer) -> int:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return 0
        saved = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        net.load_state_dict(saved["net"])
        optimizer.load_state_dict(saved["optimizer"])
        self.training_seconds = saved["seconds"]
        LOGGER.info("Resuming after epoch %d", saved["epochs_done"])
        epochs_done: int = saved["epochs_done"]
        return epochs_done

    def _save(self, net: TwoTowerNet, optimizer: torch.optim.Optimizer, epochs_done: int) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Writing then renaming means a session that dies mid-save leaves the last good file.
        temporary = self.checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epochs_done": epochs_done,
                "seconds": self.training_seconds,
            },
            temporary,
        )
        os.replace(temporary, self.checkpoint_path)


def _layers(size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size))


def _mean_of_present(vectors: Tensor, inputs: Tensor) -> Tensor:
    """Average the vectors of real entries. A row of only padding averages to zero."""
    is_present = (inputs != PADDING).unsqueeze(-1)
    return (vectors * is_present).sum(dim=1) / is_present.sum(dim=1).clamp(min=1)


def _batch_count(n_rows: int) -> int:
    return max(1, -(-n_rows // EXPORT_BATCH_SIZE))
