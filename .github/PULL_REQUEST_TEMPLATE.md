## Description

<!-- A concise explanation of what this PR does and why. The PR title alone should convey the "what"; use this section for the "why". -->

## Related issues / discussions

<!-- Link all relevant issues and discussions: "Closes #NNN", "Related to #NNN" -->

## Checklist

**General**
- [ ] The branch is rebased on the latest `main`
- [ ] Commit messages are clear and descriptive
- [ ] `pre-commit run --all-files` passes (black, isort, flake8, mypy)
- [ ] Tests cover the changes (`mamba run -n blackjax python -m pytest tests/`)

**Code quality**
- [ ] Public functions have docstrings following the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [ ] Naming follows existing conventions (`logdensity`, `jax.tree.map`, `jax.random.key()`, `jnp.clip(min=, max=)`)
- [ ] All new code is JIT-compatible

**New sampler / algorithm** *(skip if not applicable)*
- [ ] There is an open issue discussing this algorithm (use the [sampler proposal template](https://github.com/blackjax-devs/blackjax/issues/new?template=sampler_proposal.yml))
- [ ] Follows the three-layer pattern: `init` / `build_kernel` / `as_top_level_api`
- [ ] Registered in `blackjax/__init__.py`
- [ ] An example notebook has been added or updated

Consider opening a **Draft PR** first if you want early feedback on the design.
