## What changed and why

<!-- One paragraph. What does this PR do, and what problem does it solve? Link to the issue if there is one (Closes #123). -->

## Type of change

- [ ] Bug fix
- [ ] Detection improvement (new layer, threshold adjustment, pattern update)
- [ ] New feature
- [ ] Benchmark / evaluation change
- [ ] Documentation only
- [ ] Refactor (no behaviour change)

## Testing

<!-- Describe how you verified this change works correctly. -->

**Backend starts cleanly:**
- [ ] `uvicorn app.main:app --reload` — no errors on startup

**Detection layer changes** (skip if not applicable):
- [ ] Smoke test output included below
- [ ] False positive check on benign English included below

<details>
<summary>Smoke test output</summary>

```
# paste output here
```

</details>

**Frontend changes** (skip if not applicable):
- [ ] `npm run build` — no errors

## Checklist

- [ ] My changes are focused — one concern per PR
- [ ] I have not hardcoded API keys or secrets
- [ ] If I added benchmark prompts, I re-ran `evaluation/datasets/freeze_benchmarks.py`
- [ ] If I changed a detection layer, I updated the layer description in `specialist.py`
