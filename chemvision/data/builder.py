"""DatasetBuilder — orchestrates the full data construction pipeline.

Flow
----
1. If ``config.synthetic`` is set, run :class:`~chemvision.data.synthetic.SyntheticGenerator`
   on each configured simulation file.
2. If ``config.scraper`` is set, run :class:`~chemvision.data.scraper.LiteratureScraper`
   on each configured DOI / arXiv identifier.
3. If neither sub-pipeline is configured, fall back to the raw-image collect +
   annotate path (``collect_images`` / ``annotate``).
4. Shuffle + split into train / val / test according to ``config`` ratios.
5. ``save()`` serialises to HuggingFace ``DatasetDict`` format on disk.
"""

from __future__ import annotations

import random
from pathlib import Path

from chemvision.data.schema import DatasetConfig, ImageRecord

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
)


class DatasetBuilder:
    """Build, validate, and split a ChemVision dataset.

    Usage (synthetic pipeline)
    --------------------------
    >>> from chemvision.data.schema import DatasetConfig, SyntheticConfig
    >>> cfg = DatasetConfig(
    ...     source_dir=Path("raw/"),
    ...     output_dir=Path("data/processed/"),
    ...     synthetic=SyntheticConfig(files=[Path("OUTCAR"), Path("lammps.dump")]),
    ... )
    >>> builder = DatasetBuilder(cfg)
    >>> splits = builder.build()
    >>> builder.save(splits)

    Usage (literature pipeline)
    ---------------------------
    >>> from chemvision.data.schema import ScraperConfig
    >>> cfg = DatasetConfig(
    ...     source_dir=Path("raw/"),
    ...     output_dir=Path("data/processed/"),
    ...     scraper=ScraperConfig(identifiers=["2301.00001", "10.1038/s41586-023-00001-0"]),
    ... )
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_images(self) -> list[Path]:
        """Recursively collect image file paths from ``config.source_dir``.

        Filters by ``config.domains`` subdirectory names when set.

        Returns
        -------
        list[Path]
            Sorted list of image paths found under ``source_dir``.
        """
        source_dir = self.config.source_dir
        if not source_dir.exists():
            return []

        all_paths = sorted(
            p
            for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        # Filter by domain subdirectory if requested
        if self.config.domains:
            allowed = {d.value for d in self.config.domains}
            all_paths = [
                p for p in all_paths if any(part in allowed for part in p.parts)
            ]

        return all_paths

    def annotate(self, image_path: Path) -> ImageRecord:
        """Produce an :class:`ImageRecord` for a single image.

        This stub is intended for human-label or third-party annotation
        pipelines.  Subclass :class:`DatasetBuilder` and override this
        method to plug in your own annotator.

        Raises
        ------
        NotImplementedError
            Always — override in a subclass.
        """
        raise NotImplementedError(
            "annotate() must be implemented in a subclass or replaced by a "
            "sub-pipeline (synthetic / scraper) in DatasetConfig."
        )

    def build(self) -> dict[str, list[ImageRecord]]:
        """Run the full pipeline and return train / val / test splits.

        Dispatch order:

        1. Synthetic generator (if ``config.synthetic`` is set).
        2. Literature scraper (if ``config.scraper`` is set).
        3. Fallback: collect raw images and call ``annotate()``.

        Returns
        -------
        dict[str, list[ImageRecord]]
            Keys ``"train"``, ``"val"``, ``"test"``.
        """
        all_records: list[ImageRecord] = []

        # ---- synthetic pipeline ------------------------------------------
        if self.config.synthetic is not None:
            from chemvision.data.synthetic import SyntheticGenerator

            syn_cfg = self.config.synthetic
            gen = SyntheticGenerator(seed=syn_cfg.seed)
            images_dir = self.config.output_dir / "images" / syn_cfg.output_subdir

            for file_path in syn_cfg.files:
                try:
                    records = gen.generate(
                        file_path=Path(file_path),
                        output_dir=images_dir,
                        n_questions_per_difficulty=syn_cfg.n_questions_per_difficulty,
                    )
                    all_records.extend(records)
                except Exception as exc:
                    print(f"[DatasetBuilder] Skipping synthetic file {file_path}: {exc}")

        # ---- literature scraper pipeline ---------------------------------
        if self.config.scraper is not None:
            from chemvision.data.scraper import LiteratureScraper

            scr_cfg = self.config.scraper
            scraper = LiteratureScraper(
                output_dir=self.config.output_dir / "images" / scr_cfg.output_subdir,
                api_key=scr_cfg.api_key,
                max_figures_per_paper=scr_cfg.max_figures_per_paper,
                request_delay=scr_cfg.request_delay,
                include_image_in_haiku_call=scr_cfg.include_image_in_haiku_call,
            )
            try:
                records = scraper.scrape(scr_cfg.identifiers)
                all_records.extend(records)
            except Exception as exc:
                print(f"[DatasetBuilder] Scraper pipeline failed: {exc}")

        # ---- fallback: raw image collect + annotate ----------------------
        if not all_records:
            for image_path in self.collect_images():
                try:
                    all_records.append(self.annotate(image_path))
                except NotImplementedError:
                    raise
                except Exception as exc:
                    print(f"[DatasetBuilder] Skipping {image_path}: {exc}")

        return self._split(all_records)

    def save(self, splits: dict[str, list[ImageRecord]]) -> None:
        """Serialise splits to a HuggingFace ``DatasetDict`` on disk.

        The dataset is saved with ``DatasetDict.save_to_disk()`` to
        ``config.output_dir / "hf_dataset"``.  Image paths are stored as
        strings so the dataset is portable across machines.

        Parameters
        ----------
        splits:
            Dict returned by :meth:`build`.
        """
        try:
            from datasets import Dataset, DatasetDict
        except ImportError as exc:
            raise ImportError(
                "datasets is required for HuggingFace format. "
                "Install with: pip install datasets"
            ) from exc

        def _to_row(record: ImageRecord) -> dict:
            # model_dump(mode='json') coerces Path → str and Literal → str
            row = record.model_dump(mode="json")
            return row

        ds_dict = DatasetDict(
            {
                split_name: Dataset.from_list([_to_row(r) for r in records])
                for split_name, records in splits.items()
                if records  # skip empty splits
            }
        )

        save_path = self.config.output_dir / "hf_dataset"
        save_path.mkdir(parents=True, exist_ok=True)
        ds_dict.save_to_disk(str(save_path))
        total = sum(len(v) for v in splits.values())
        print(
            f"[DatasetBuilder] Saved {total} records to {save_path}\n"
            + "\n".join(f"  {k}: {len(v)}" for k, v in splits.items() if v)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
        """Shuffle and split records into train / val / test."""
        rng = random.Random(self.config.seed)
        shuffled = records.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
