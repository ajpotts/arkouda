
# Arkouda v2025.X.X

We're excited to announce a feature-packed release of Arkouda with enhanced NumPy compatibility, powerful new array functions, performance improvements, CI tooling, and major documentation progress.

---

## New Features

- Simplify offline builds ([#3957](https://github.com/Bears-R-Us/arkouda/issues/3957), [#3944](https://github.com/Bears-R-Us/arkouda/issues/3944), [PR #3958](https://github.com/Bears-R-Us/arkouda/pull/3958))
- `skip_by_rank` enhanced for set containment ([#3954](https://github.com/Bears-R-Us/arkouda/issues/3954))
- Read multiple Parquet row groups correctly ([PR #3989](https://github.com/Bears-R-Us/arkouda/pull/3989))
- Removed duplicate Dockerfile ([#3976](https://github.com/Bears-R-Us/arkouda/issues/3976))
- Upgraded to Apache Arrow 19.0.0 ([#3981](https://github.com/Bears-R-Us/arkouda/issues/3981))
- Added `ak.tile`, `ak.repeat`, `ak.take`, `ak.diff`, `ak.nextafter`, `ak.newaxis`, `ak.quantile`, `ak.percentile`, `ak.vecdot`, `ak.result_type`, `ak.eye`, and `ak.append` ([#3000](https://github.com/Bears-R-Us/arkouda/issues/3000), [#2998](https://github.com/Bears-R-Us/arkouda/issues/2998), [#3004](https://github.com/Bears-R-Us/arkouda/issues/3004), [#3292](https://github.com/Bears-R-Us/arkouda/issues/3292), [#4483](https://github.com/Bears-R-Us/arkouda/issues/4483), [#4502](https://github.com/Bears-R-Us/arkouda/issues/4502))
- Introduced `ak.apply` ([#3963](https://github.com/Bears-R-Us/arkouda/issues/3963))
- Server functions `repartitionByLocaleString` and `repartitionByHashString` added ([#4497](https://github.com/Bears-R-Us/arkouda/issues/4497), [#4499](https://github.com/Bears-R-Us/arkouda/issues/4499))
- Experimental checkpointing of server state ([#2384](https://github.com/Bears-R-Us/arkouda/issues/2384))

---

## API Enhancements and Compatibility

- NumPy 2.0 compatibility: `can_cast`, `sign`, `result_type`, `dtype`, etc. ([#3337](https://github.com/Bears-R-Us/arkouda/issues/3337), [#4098](https://github.com/Bears-R-Us/arkouda/issues/4098), [#4555](https://github.com/Bears-R-Us/arkouda/issues/4555))
- Shape-related: `ak.reshape`, `ak.transpose`, `ak.arange`, `ak.full` enhancements ([#4165](https://github.com/Bears-R-Us/arkouda/issues/4165), [#4092](https://github.com/Bears-R-Us/arkouda/issues/4092), [#4321](https://github.com/Bears-R-Us/arkouda/issues/4321), [#4312](https://github.com/Bears-R-Us/arkouda/issues/4312))
- Added `where` parameter to many functions ([#4520](https://github.com/Bears-R-Us/arkouda/issues/4520))
- `coargsort` now supports `ascending` keyword ([#4464](https://github.com/Bears-R-Us/arkouda/issues/4464))

---

## DataFrame and Merge Improvements
- `comm_diagnostics` now returns a DataFrame ([#3970](https://github.com/Bears-R-Us/arkouda/issues/3970), [PR #3971](https://github.com/Bears-R-Us/arkouda/pull/3971))

- Added `right_on` and `left_on` to `DataFrame.merge` ([#4234](https://github.com/Bears-R-Us/arkouda/issues/4234))
- `ak.merge` supports `Categorical` ([#4313](https://github.com/Bears-R-Us/arkouda/issues/4313))
- Fixed `DataFrame.__getitem__` behavior during merges ([#4360](https://github.com/Bears-R-Us/arkouda/issues/4360))

---

## Bug Fixes

- Fixed permutation instability ([#3974](https://github.com/Bears-R-Us/arkouda/issues/3974))
- Fixed `ak.ceil`, `ak.trunc`, `ak.permutation`, `ak.full` errors ([#3984](https://github.com/Bears-R-Us/arkouda/issues/3984), [#4312](https://github.com/Bears-R-Us/arkouda/issues/4312))
- Resolved segmented search crash ([PR #4367](https://github.com/Bears-R-Us/arkouda/pull/4367))
- Resolved issues with `max_bits`, `searchsorted`, and `repartition` functions ([#4173](https://github.com/Bears-R-Us/arkouda/issues/4173), [#4203](https://github.com/Bears-R-Us/arkouda/issues/4203), [#4235](https://github.com/Bears-R-Us/arkouda/issues/4235))

---

## Performance Improvements
- Sort module updates for Chapel 2.3 ([PR #3972](https://github.com/Bears-R-Us/arkouda/pull/3972))

- Refactored benchmarks: `stream`, `scatter`, `bitwise`, `array_create`, `gather`, etc. ([#3580](https://github.com/Bears-R-Us/arkouda/issues/3580), [#3561](https://github.com/Bears-R-Us/arkouda/issues/3561), [#3563](https://github.com/Bears-R-Us/arkouda/issues/3563))
- Introduced `where` benchmark ([#4581](https://github.com/Bears-R-Us/arkouda/issues/4581))
- Improved launch scripts for PBS environments ([PR #4088](https://github.com/Bears-R-Us/arkouda/pull/4088))

---

## Documentation

- Doctest coverage added across `strings`, `categorical`, `numeric`, `groupbyclass`, `array_api`, and more ([#4282](https://github.com/Bears-R-Us/arkouda/issues/4282), [#4274](https://github.com/Bears-R-Us/arkouda/issues/4274), [#4291](https://github.com/Bears-R-Us/arkouda/issues/4291))
- Removed pydocstyle and darglint errors: `DAR102`, `DAR202`, `D400`, `D205`, `DOC602`, etc. ([#4352](https://github.com/Bears-R-Us/arkouda/issues/4352), [#4355](https://github.com/Bears-R-Us/arkouda/issues/4355), [#4339](https://github.com/Bears-R-Us/arkouda/issues/4339))
- Numpy-style modules reorganized into dedicated directories (`numpy`, `pandas`, `scipy`) ([PR #4183](https://github.com/Bears-R-Us/arkouda/pull/4183))

---

## Developer Experience

- CI updated for Python 3.13, Chapel 2.3, and better dependency reuse ([#4085](https://github.com/Bears-R-Us/arkouda/issues/4085), [#4228](https://github.com/Bears-R-Us/arkouda/issues/4228))
- Added: `flake8`, `ruff`, `darglint`, `pydocstyle`, `pydoclint`, `docstr-coverage`, `chplcheck`, `pytest-timeout` ([#3980](https://github.com/Bears-R-Us/arkouda/issues/3980), [#4113](https://github.com/Bears-R-Us/arkouda/issues/4113), [#4402](https://github.com/Bears-R-Us/arkouda/issues/4402), [#4386](https://github.com/Bears-R-Us/arkouda/issues/4386), [#4388](https://github.com/Bears-R-Us/arkouda/issues/4388), [#4445](https://github.com/Bears-R-Us/arkouda/issues/4445))
- Git pre-commit hooks added ([#4356](https://github.com/Bears-R-Us/arkouda/issues/4356))
- Makefile additions: `make format`, `make ruff-format`, `make isort`, `install-pytables` ([#4206](https://github.com/Bears-R-Us/arkouda/issues/4206), [#4211](https://github.com/Bears-R-Us/arkouda/issues/4211), [#4200](https://github.com/Bears-R-Us/arkouda/issues/4200))

---

## New Contributors

- [@1RyanK](https://github.com/1RyanK) – multiple core features, DataFrame fixes, testing infrastructure, and doc coverage
- [@alvaradoo](https://github.com/alvaradoo) – environment improvements and CI

---

## Full Changelog

[Compare full changes from v2025.01.13 to this release](https://github.com/Bears-R-Us/arkouda/compare/v2025.01.13...v2025)
