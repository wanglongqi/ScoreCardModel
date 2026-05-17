# Contributing to ScoreCardModel

Thank you for considering contributing to ScoreCardModel! We welcome contributions in the form of bug reports, feature requests, documentation improvements, and code changes.

## Development Setup

We use `uv` for dependency management. To get started:

1. Clone the repository.
2. Install dependencies: `uv sync --all-extras`
3. Install pre-commit hooks (if applicable).

## Testing

Run tests using `pytest`:

```bash
uv run pytest
```

Ensure all tests pass before submitting a pull request. We aim for high test coverage, especially for core algorithmic components.

## Documentation

Documentation is built with MkDocs. To preview changes locally:

```bash
uv run mkdocs serve
```

## Pull Request Process

1. Create a new branch for your changes.
2. Implement your changes and add tests.
3. Ensure the code follows our style guidelines (run `ruff check .`).
4. Update the documentation if necessary.
5. Submit a pull request with a clear description of the changes.

## Coding Standards

- Follow PEP 8.
- Use type hints where appropriate.
- Write clear, concise docstrings (Google style preferred).
- Keep functions small and focused.
