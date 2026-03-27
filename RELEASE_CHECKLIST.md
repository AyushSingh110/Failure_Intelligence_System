# Release Checklist

Use this checklist every time you publish a new `fie-sdk` version.

## 1. Update package metadata

- Bump version in `pyproject.toml`
- Bump version in `fie/__init__.py`
- Update README sections that mention the current package version
- Review package description, keywords, and URLs in `pyproject.toml`

## 2. Make sure the README is ready for PyPI

PyPI shows the package long description from `README.md` because `pyproject.toml` sets:

```toml
readme = { file = "README.md", content-type = "text/markdown" }
```

That means the PyPI package page only updates when a new package version is uploaded.

Before publishing:

- Confirm `README.md` reflects the latest features
- Make sure the package section mentions the correct version
- Avoid broken relative links or repo-only assumptions where possible

## 3. Clean old build artifacts

Delete old files from `dist/` before building a new release.

Windows PowerShell:

```powershell
Remove-Item dist\* -Force
```

## 4. Build the package

Install release tooling if needed:

```bash
pip install build twine
```

Build:

```bash
python -m build
```

Expected output:

- `dist/fie_sdk-<version>-py3-none-any.whl`
- `dist/fie_sdk-<version>.tar.gz`

## 5. Validate the package

```bash
python -m twine check dist/*
```

Optional local install test:

```bash
pip install --force-reinstall dist/fie_sdk-<version>-py3-none-any.whl
```

## 6. Publish to PyPI

Upload manually:

```bash
python -m twine upload dist/*
```

Or use the GitHub Actions workflow in `.github/workflows/publish-pypi.yml`.

For the GitHub Action to publish successfully, configure PyPI trusted publishing for this repository, or switch the workflow to a token-based publish setup.

## 7. Verify the release

- Open the PyPI package page
- Confirm the version is the new one
- Confirm the README rendered correctly on PyPI
- Run:

```bash
pip install --upgrade fie-sdk
```

- Check that installing from PyPI now resolves to the new version

## 8. Tag and document the release

- Create a Git tag such as `v0.2.0`
- Push the tag
- Add release notes on GitHub
- Mention major additions like explainability, auth, dashboard changes, and deployment support
