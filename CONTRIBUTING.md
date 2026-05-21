# Contributing to Failure Intelligence Engine

Thank you for your interest in contributing to FIE. This document explains how to get started, how to report issues, and how to submit changes.

---

## Ways to Contribute

- **Bug reports** — found something broken? Open an issue.
- **Feature suggestions** — have an idea? Open a discussion or issue.
- **Code contributions** — fix a bug, improve performance, or add a feature via a pull request.
- **Documentation** — improve the README, add examples, or fix typos.

---

## Getting Started

### 1. Fork and clone the repository

```bash
git clone https://github.com/AyushSingh110/Failure_Intelligence_System.git
cd Failure_Intelligence_System
```

### 2. Set up the backend

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your GROQ_API_KEY, MONGODB_URI, etc.
uvicorn app.main:app --reload --port 8000
```

### 3. Set up the frontend

```bash
cd Frontend
npm install
cp .env.local.example .env.local   # fill in VITE_API_URL, VITE_GOOGLE_CLIENT_ID
npm run dev
```

---

## Reporting Bugs

Before opening an issue, please:

1. Search existing issues to avoid duplicates.
2. Include the following in your report:
   - What you did (steps to reproduce)
   - What you expected to happen
   - What actually happened
   - Relevant logs or error messages
   - Your environment (Python version, OS, fie-sdk version)

---

## Submitting a Pull Request

1. Create a branch from `main` with a descriptive name:
   ```bash
   git checkout -b fix/preflight-false-positive
   ```

2. Make your changes. Keep them focused — one PR per concern.

3. Make sure the backend still starts cleanly:
   ```bash
   uvicorn app.main:app --reload
   ```

4. Make sure the frontend builds without errors:
   ```bash
   cd Frontend && npm run build
   ```

5. Open a PR against `main` with a clear title and description of what changed and why.

---

## Code Style

- **Python** — follow PEP 8. Use type hints where practical. Keep functions small and focused.
- **JavaScript / React** — functional components only, no class components. Inline styles to match the existing design system.
- **Comments** — only when the *why* is non-obvious. Do not describe what the code does.

---

## Project Structure

```
app/
  routes/         FastAPI route modules (one domain per file)
  auth.py         User management and JWT sessions
  schemas.py      Pydantic request/response models

engine/
  agents/         DiagnosticJury specialist agents
  detector/       Signal detectors (entropy, consistency, embedding)
  groq_service.py Shadow model fan-out via Groq API

Frontend/src/
  pages/          One file per dashboard page
  components/     Shared UI components
  lib/            Auth utilities

fie/              (Published as fie-sdk on PyPI — separate package)
```

---

## Questions

If you are unsure about anything, open a GitHub Discussion or email [ayushsingh355vns@gmail.com](mailto:ayushsingh355vns@gmail.com). No question is too small.
