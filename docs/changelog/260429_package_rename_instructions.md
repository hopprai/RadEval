# `RadEval` → `radeval` rollout — remaining TODOs

The in-tree rename landed in the `package-rename` PR. These items are
outside this repo and need a human decision / access I don't have.
Full context in `docs/changelog/260429.md`.

## Required

### 1. PyPI: publish `radeval` 2.2.0

The old `RadEval` distribution is frozen at 2.1.0 on PyPI. To
release 2.2.0 under the new name:

```bash
# On a clean checkout of main, after this PR merges:
python -m build                     # produces dist/radeval-2.2.0.{tar.gz,whl}
twine upload dist/radeval-2.2.0*    # requires a radeval project on PyPI
```

If `radeval` isn't already registered on PyPI, the first `twine
upload` will create the project. Confirm the namespace isn't squatted
before uploading.

### 2. Deprecate `RadEval` on PyPI

Update the old `RadEval` project's PyPI description (or push a 2.1.1
metadata-only release) with a one-line pointer:

> This package has been renamed to `radeval`. Install with
> `pip install radeval`. See
> https://github.com/jbdel/RadEval/blob/main/docs/changelog/260429.md

Don't yank 2.1.0 — existing `RadEval==2.1.0` pins should keep working.

### 3. Sync the public repo

`scripts/publish_public.py --push -m "..."` from a shell with
credentials. This mirrors the renamed package tree into
`jbdel/RadEval`. Verify the dry-run first (no `--push`) — should
show `Removed radeval/...` for private metrics and `Leak scan: 0
unexpected hits`.

### 4. Announce the break

External users with `RadEval` in a requirements file will need to
update. Good places to mention:

- GitHub Release notes for `v2.2.0` (call out the install + import
  change up front).
- README.

Migration is mechanical:

```diff
- pip install RadEval
+ pip install radeval
```
```diff
- from RadEval import RadEval
+ from radeval import RadEval
```

## Optional decisions

### 5. Rename the GitHub repos?

The changelog intentionally left `jbdel/RadEval` and `hopprai/RadEval`
named `RadEval` — the brand. GitHub auto-redirects old URLs, so
renaming the repo to `radeval` is safe but not required. My
recommendation: **leave as `RadEval`** for brand continuity with the
paper and HF model IDs (`IAMJB/RadEvalModernBERT`). If you do rename,
update:

- Repo name under Settings → General.
- Any CI configs, Dependabot configs, docs site deploy keys that
  hardcode the URL.
- Badge URLs in the README.
- The `PUBLIC_REMOTE` constant in `scripts/publish_public.py`.

### 6. HF model IDs

The `IAMJB/RadEvalModernBERT` model stays named `RadEvalModernBERT`
(brand). No action needed unless IAMJB wants to rename it separately.
