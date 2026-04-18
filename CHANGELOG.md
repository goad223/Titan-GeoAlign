# Changelog

All notable changes to **Titan-GeoAlign** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (nothing yet)

### Changed
- (nothing yet)

### Deprecated
- (nothing yet)

### Removed
- (nothing yet)

### Fixed
- (nothing yet)

### Security
- (nothing yet)

---

## [0.1.0] — 2024-10-01

### Added

#### Project Foundation
- Initial project scaffold: `pyproject.toml`, `Makefile`, `environment.yml`.
- `requirements.txt`, `requirements-dev.txt`, `requirements-gpu.txt` with pinned dependencies.
- `.gitignore`, `.gitattributes`, `.editorconfig` for consistent development environment.
- `.pre-commit-config.yaml` with black, ruff, mypy, bandit, detect-secrets, yamllint, markdownlint, and conventional-commit hooks.
- Apache 2.0 `LICENSE`.
- `CONTRIBUTING.md` with fork-branch-PR workflow, code style guide, testing requirements, and Conventional Commits format.
- `CODE_OF_CONDUCT.md` based on Contributor Covenant 2.1.
- `SECURITY.md` with responsible disclosure policy and coordinated disclosure timeline.
- `CHANGELOG.md` (this file).

#### Core Architecture (planned for 0.2.0)
- Geospatial image registration pipeline using deep feature matching.
- Sub-pixel alignment via phase correlation and optical flow.
- Multi-scale pyramid processing for large satellite imagery.
- GDAL/rasterio integration with full CRS-aware reprojection.
- FastAPI REST endpoint `/v1/register` for batch alignment jobs.
- gRPC service definition for high-throughput inference.
- Ray-based distributed job scheduler for large mosaics.
- MLflow experiment tracking integration.
- DVC pipeline for reproducible data workflows.
- Structured logging with `structlog` and OpenTelemetry traces.
- Prometheus metrics endpoint for production monitoring.

### Dependencies
- PyTorch 2.5.0 with CUDA 12.4 support.
- Hugging Face transformers 4.45.0 + diffusers 0.30.0 foundation model backbone.
- GDAL 3.8.4 / rasterio 1.3.10 / pyproj 3.6.1 / shapely 2.0.3 geospatial stack.
- kornia 0.7.3 for differentiable geometric computer vision.
- AROSICS 1.10.3 for sub-pixel co-registration.
- FastAPI 0.115.0 + pydantic 2.9.0 for the API layer.
- Hydra-Core 1.3.2 + OmegaConf 2.3.0 for hierarchical configuration.
- Ray 2.36.0 + Dask 2024.9.0 for distributed processing.

---

[Unreleased]: https://github.com/goad223/Titan-GeoAlign/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/goad223/Titan-GeoAlign/releases/tag/v0.1.0
