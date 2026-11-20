# Archive backup — 20 November 2026

This branch preserves the local unilateral_v2 commit and the selected mnnpy revision (6083c25). The submodule URL points to the fork where that revision was verified on GitHub.

Additional preserved refs in this repository:
- `backup/archive-2026-11-20-unilateral-v3`: the previously local-only unilateral_v3 branch.
- `backup/archive-2026-11-20-stash`: the original stash commit, including its index and parent history. To recover it, check out its first parent in a new branch, then run `git stash apply origin/backup/archive-2026-11-20-stash`.

The original local stash was retained. Python environments and caches are not included.
