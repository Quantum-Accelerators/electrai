# Spec: Git History Cleanup — Remove Data Blobs

## Problem

The repo's `.git/objects/pack/` is ~978MB, largely due to historical data
file commits under `data/MP/`:
- `39933b1` committed 10 CHGCARs + 10 json.gz label files (~1.1M lines)
- `ce5f8e8` ("Trim example data #59") removed some, but blobs persist
- Earlier commits may have had even larger files

Current HEAD no longer contains these files (removed in `d093d96`), but
every clone still downloads the full history including old blobs.

## Goal

Produce a clean `main` branch where data blobs never existed in history,
so fresh clones are small (~50MB instead of ~1GB).

## Approach

### 1. Preserve current history

Keep a branch (e.g. `pre-cleanup` or `archive/with-data-blobs`) pointing
at the current `main` HEAD before rewriting. Push to the OA fork (`oa`
remote) for archival — doesn't need to live on the public `q` remote.

### 2. Rewrite history with `git filter-repo`

```bash
# Install if needed
pip install git-filter-repo

# Dry run: identify large blobs
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $2 > 1000000 {print $2, $3}' \
  | sort -rn | head -20

# Rewrite: remove data dirs from all commits
git filter-repo \
  --path data/MP/chgcars/ --path data/MP/jsongz/ \
  --invert-paths \
  --force
```

This removes the paths from every commit, rewriting SHAs. The
`.gitignore` files we added in `d093d96` will survive (they're in
`data/MP/chgcars/.gitignore`, not under `input/` or `label/`).

Wait — `--path data/MP/chgcars/` would also remove the `.gitignore`
we just added there. Options:
- a) Run filter-repo with more specific paths (each blob file)
- b) Re-add the `.gitignore` files after rewriting
- c) Use `--path-glob 'data/MP/chgcars/**/*.CHGCAR'` etc.

Option (c) is cleanest:
```bash
git filter-repo \
  --path-glob 'data/MP/chgcars/**/*.CHGCAR' \
  --path-glob 'data/MP/jsongz/**/*.json.gz' \
  --invert-paths \
  --force
```

### 3. Verify

```bash
# Check pack size after rewriting
git gc --aggressive --prune=now
du -sh .git/objects/pack/

# Verify no data blobs remain
git rev-list --objects --all \
  | git cat-file --batch-check='%(objectsize) %(rest)' \
  | awk '$1 > 5000000' | head

# Run e2e test (needs dvx pull first)
dvx pull s3/openathena/electrai/{input,label}/mp-{1775579,1828106,1828986,1887555,1924667}.CHGCAR.dvc
python tests/e2e_train.py -v
```

### 4. Force push

```bash
# Push rewritten main to public upstream
git push q main --force-with-lease

# Push archive branch to OA fork only
git push oa pre-cleanup
```

All collaborators will need to re-clone or `git fetch --all && git reset
--hard q/main` after the force push.

## Risks

- **Open PRs**: Any open PRs will have base SHAs that no longer exist.
  They'll need to be rebased onto the new history.
- **CI caches**: GHA caches keyed on SHAs will miss. Not a real problem,
  just slower first runs.
- **Local clones**: Everyone needs to re-clone or hard-reset. Announce in
  Slack before doing this.

## Timing

Not urgent — the data is already removed from HEAD. This only matters for
clone size. Can be done whenever convenient, ideally during a quiet period
with no open PRs.
