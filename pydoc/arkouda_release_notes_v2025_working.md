

# Arkouda v2025.07.03



We're excited to announce a feature-packed release of Arkouda with enhanced NumPy compatibility, powerful new array functions, performance improvements, CI tooling, and major documentation progress.



---

## Features

**Array Functions**

- **Added**: `append`, `argsort`, `diff`, `eye`, `newaxis`, `nextafter`, `percentile`, `quantile`, `repeat`, `result_type`, `take`, `tile`, `vecdot`, `xp.trapz`  
  (#2998, #3000, #3003, #3004, #3292, #3755, #4393, #4418, #4419, #4458, #4483, #4484, #4502, [PR #4101](https://github.com/Bears-R-Us/arkouda/pull/4101), [PR #4127](https://github.com/Bears-R-Us/arkouda/pull/4127), [PR #4146](https://github.com/Bears-R-Us/arkouda/pull/4146), [PR #4219](https://github.com/Bears-R-Us/arkouda/pull/4219), [PR #4361](https://github.com/Bears-R-Us/arkouda/pull/4361), [PR #4393](https://github.com/Bears-R-Us/arkouda/pull/4393)), [PR #4394](https://github.com/Bears-R-Us/arkouda/pull/4394), [PR #4418](https://github.com/Bears-R-Us/arkouda/pull/4418), [PR #4419](https://github.com/Bears-R-Us/arkouda/pull/4419), [PR #4552](https://github.com/Bears-R-Us/arkouda/pull/4552))

- **Improved**: `ak.diff`, `ak.nextafter`, `ak.repeat`, `ak.reshape`, `ak.take`, `ak.tile`, `ak.argsort`  
  (#2998, #3000, #3004, #3755, #4101, #4146, #4147, #4165, #4394, #4418, #4419, #4458, [PRs #4101](https://github.com/Bears-R-Us/arkouda/pull/4101), [#4146](https://github.com/Bears-R-Us/arkouda/pull/4146), [#4394](https://github.com/Bears-R-Us/arkouda/pull/4394), [#4552](https://github.com/Bears-R-Us/arkouda/pull/4552))

- **Axis and Broadcasting Enhancements**:
  - Axis support in `ak.mean`, `ak.var`, `ak.std` (#4425, [PR #4442](https://github.com/Bears-R-Us/arkouda/pull/4442))  
  - Negative axis handling in `ak.squeeze`, `ak.repeat`, `ak.argmin`, `ak.argmax` (#4407, #4421, [PR #4406](https://github.com/Bears-R-Us/arkouda/pull/4406), [PR #4408](https://github.com/Bears-R-Us/arkouda/pull/4408))


**Checkpointing and Logging**

- Introduced experimental checkpointing of server state, with support for numeric arrays and automatic checkpointing triggered by memory limits or idle time.  
  (#2384, [PRs #3915](https://github.com/Bears-R-Us/arkouda/pull/3915), [#4391](https://github.com/Bears-R-Us/arkouda/pull/4391), [#4549](https://github.com/Bears-R-Us/arkouda/pull/4549), [#4644](https://github.com/Bears-R-Us/arkouda/pull/4644))

- Improved logging behavior:
  - Logs can now be redirected to a file using the server’s logging mechanism ([PR #4152](https://github.com/Bears-R-Us/arkouda/pull/4152))
  - Reduced use of `throws` in logging routines ([PR #4433](https://github.com/Bears-R-Us/arkouda/pull/4433))

**Project Infrastructure**

- Upgraded Apache Arrow to 19.0.1 for compatibility and stability improvements  
  (#3981, [PRs #3982](https://github.com/Bears-R-Us/arkouda/pull/3982), [#4342](https://github.com/Bears-R-Us/arkouda/pull/4342))


**Other**

- Introduced `ak.apply`, `ak.result_type` (now with `bigint` support), and `ak.searchsorted`  
  (#3005, #4483, [PRs #3963](https://github.com/Bears-R-Us/arkouda/pull/3963), [#4214](https://github.com/Bears-R-Us/arkouda/pull/4214), [#4484](https://github.com/Bears-R-Us/arkouda/pull/4484))

- Added `ak.coargsort(ascending=...)` keyword argument  
  (#4464, [PR #4467](https://github.com/Bears-R-Us/arkouda/pull/4467))

- Added standard gamma distribution function to `ak.random`  
  (#3846, [PR #4089](https://github.com/Bears-R-Us/arkouda/pull/4089))


---


## API Enhancements and Compatibility


**API Enhancements and Compatibility**

- Improved NumPy 2.0 compatibility:
  - Upgraded numpy dependency to 2.0.0 (#4098, [PR #4188](https://github.com/Bears-R-Us/arkouda/pull/4188), [PR #4213](https://github.com/Bears-R-Us/arkouda/pull/4213)) 
  - Added or aligned: `ak.can_cast`, `ak.sign`, `ak.result_type`, `ak.dtype`, `ak.vecdot`, `ak.eye`, `ak.dot`, `ak.arange`, `ak.transpose`, `ak.hstack`, `ak.where`, `ak.full`, `ak.reshape` 
  (#3329,  #3337, #4092, #4165, #4312, #4321, #4555, #4468, [PR #4105](https://github.com/Bears-R-Us/arkouda/pull/4105), [PR #4116](https://github.com/Bears-R-Us/arkouda/pull/4116), [PR #4174](https://github.com/Bears-R-Us/arkouda/pull/4174), [PR #4224](https://github.com/Bears-R-Us/arkouda/pull/4224), [PR #4472](https://github.com/Bears-R-Us/arkouda/pull/4472), [PR #4522](https://github.com/Bears-R-Us/arkouda/pull/4522), [PR #4556](https://github.com/Bears-R-Us/arkouda/pull/4556))
  - Improved parameter alignment to NumPy (`ak.eye`, `ak.where`, `ak.histogram`, etc.) (#4096, [PR #4078](https://github.com/Bears-R-Us/arkouda/pull/4078), [PR #4482](https://github.com/Bears-R-Us/arkouda/pull/4482))
  - Enabled `bool` as alias for `bool_`; enhanced dtype detection for builtins `bool`, `float`, `int` (#4186, #4627, [PR #4187](https://github.com/Bears-R-Us/arkouda/pull/4187), [PR #4628](https://github.com/Bears-R-Us/arkouda/pull/4628))
  (#3337, #3329, #3337, #3981, #4092, #4096,  #4105, #4116, #4124, #4165, #4186, #4188, #4213, #4224, #4321, #4312, #4468, #4481, #4483, #4501, #4520, #4555, #4556, #4552, #4627; [PRs #4078](https://github.com/Bears-R-Us/arkouda/pull/4078), [#4103](https://github.com/Bears-R-Us/arkouda/pull/4103), [#4174](https://github.com/Bears-R-Us/arkouda/pull/4174), [#4185](https://github.com/Bears-R-Us/arkouda/pull/4185), [#4390](https://github.com/Bears-R-Us/arkouda/pull/4390), [#4213](https://github.com/Bears-R-Us/arkouda/pull/4213), [#4505](https://github.com/Bears-R-Us/arkouda/pull/4505), [#4522](https://github.com/Bears-R-Us/arkouda/pull/4522), [#4628](https://github.com/Bears-R-Us/arkouda/pull/4628))

- Reorganized modules into dedicated `numpy/`, `scipy/` directories for API clarity  
  ([PRs #4103](https://github.com/Bears-R-Us/arkouda/pull/4103), [#4185](https://github.com/Bears-R-Us/arkouda/pull/4185), [#4390](https://github.com/Bears-R-Us/arkouda/pull/4390))

- Miscellaneous API additions and improvements:
  - `ak.coargsort` now supports `ascending=` keyword (#4464, [PR #4467](https://github.com/Bears-R-Us/arkouda/pull/4467))
  - `comm_diagnostics` now returns a DataFrame (#3970, [PR #3971](https://github.com/Bears-R-Us/arkouda/pull/3971))

- DataFrame and Merge Improvements

  - `DataFrame.merge` now supports `left_on` and `right_on`  
  (#4234, [PR #4240](https://github.com/Bears-R-Us/arkouda/pull/4240))

  - `ak.merge` now supports merging on `Categorical` columns  
  (#4313, [PR #4344](https://github.com/Bears-R-Us/arkouda/pull/4344))

  - Fixed `DataFrame.__getitem__` dispatch behavior during merges  
  (#4360, [PR #4362](https://github.com/Bears-R-Us/arkouda/pull/4362))




---

## Performance Improvements

- Improved performance and stability in `ak.permutation`, distributed array creation, and sorting  
  (#3974, [PRs #3975](https://github.com/Bears-R-Us/arkouda/pull/3975), [#4242](https://github.com/Bears-R-Us/arkouda/pull/4242))


---

## Deprecations and Refactors

- parse_single_value to handle uints represented as nega… (#4147, [PR #4150](https://github.com/Bears-R-Us/arkouda/pull/4150))
- Change some nested foralls to for/foralls in HistogramMsg ([PR #4180](https://github.com/Bears-R-Us/arkouda/pull/4180))
- remove try! in toSymEntry (#4175, [PR #4176](https://github.com/Bears-R-Us/arkouda/pull/4176))
- Workaround an issue with using PrivateSpace domains in formals ([PR #4633](https://github.com/Bears-R-Us/arkouda/pull/4633))
- Create a repartitionByLocaleString function (#4497, [PR #4557](https://github.com/Bears-R-Us/arkouda/pull/4557))
- Create a repartitionByHashString function (#4499, [PR #4617](https://github.com/Bears-R-Us/arkouda/pull/4617))
- Server functions `repartitionByLocaleString` and `repartitionByHashString` added (#4497, #4499)
- Make an unordered set union of two Strings arrays function (#4244, [PR #4245](https://github.com/Bears-R-Us/arkouda/pull/4245))
- Add a compatibility module for Time to handle the new totalMicroseconds method ([PR #4142](https://github.com/Bears-R-Us/arkouda/pull/4142))
- refactor arkouda/numpy/imports/__init__.py ([PR #4453](https://github.com/Bears-R-Us/arkouda/pull/4453))
- Sort module updates for Chapel 2.3 ([PR #3972](https://github.com/Bears-R-Us/arkouda/pull/3972))
- Tweaks a few files to avoid relying on CHPL_HOME ([PR #4551](https://github.com/Bears-R-Us/arkouda/pull/4551))
- add missing __all__ (#4426, [PR #4427](https://github.com/Bears-R-Us/arkouda/pull/4427))


**Deprecations and Refactors**

- Removed deprecated or obsolete features:
  - Removed deprecated functions including `lookup` function and legacy server utilities  
    (#4308, #4375, [PRs #4309](https://github.com/Bears-R-Us/arkouda/pull/4309), [#4374](https://github.com/Bears-R-Us/arkouda/pull/4374), [#4376](https://github.com/Bears-R-Us/arkouda/pull/4376))
  - Old `registerND` annotations removed from remaining modules  
    (#3721, #3723, [PR #3986](https://github.com/Bears-R-Us/arkouda/pull/3986))

- Refactored and modernized core logic:
  - `ak.arange` now uses `instantiateAndRegister` (#4382, [PR #4383](https://github.com/Bears-R-Us/arkouda/pull/4383))
  - Improved logic for `binopvv` and `binopvs` (#4459, #4460, [PRs #4462](https://github.com/Bears-R-Us/arkouda/pull/4462), [#4563](https://github.com/Bears-R-Us/arkouda/pull/4563))
  - Reverted `ak.zeros` behavior to previous default ([PR #4141](https://github.com/Bears-R-Us/arkouda/pull/4141))
  - Refactored import and module layout (`__init__.py`, sort module, CHPL_HOME independence)  
    ([PRs #3972](https://github.com/Bears-R-Us/arkouda/pull/3972), [#4453](https://github.com/Bears-R-Us/arkouda/pull/4453), [#4551](https://github.com/Bears-R-Us/arkouda/pull/4551))

- Simplified internals and extended platform support:
  - Logic cleanup in `parse_single_value`, `HistogramMsg`, `toSymEntry`, and PrivateSpace domain usage  
    ([PRs #4150](https://github.com/Bears-R-Us/arkouda/pull/4150), [#4176](https://github.com/Bears-R-Us/arkouda/pull/4176), [#4180](https://github.com/Bears-R-Us/arkouda/pull/4180), [#4633](https://github.com/Bears-R-Us/arkouda/pull/4633))

- Added internal or system-level functionality:
  - `repartitionByLocaleString` and `repartitionByHashString` server functions  
    (#4497, #4499, [PRs #4557](https://github.com/Bears-R-Us/arkouda/pull/4557), [#4617](https://github.com/Bears-R-Us/arkouda/pull/4617))
  - Set union function for `Strings` arrays (#4244, [PR #4245](https://github.com/Bears-R-Us/arkouda/pull/4245))
  - Compatibility module for `Time.totalMicroseconds()` ([PR #4142](https://github.com/Bears-R-Us/arkouda/pull/4142))
  - Added missing `__all__` to ensure symbol export consistency (#4426, [PR #4427](https://github.com/Bears-R-Us/arkouda/pull/4427))



**Benchmark Refactor**

- Refactor benchmark handling of running mode. (#3964, [PR #4471](https://github.com/Bears-R-Us/arkouda/pull/4471))
- Part of #3964:  Refactor benchmark handling of running mode ([PR #4358](https://github.com/Bears-R-Us/arkouda/pull/4358))
- Part 2 of #3964:  Refactor benchmark handling of running mode ([PR #4373](https://github.com/Bears-R-Us/arkouda/pull/4373))
- Part 3 of #3964:  Refactor benchmark handling of running mode ([PR #4385](https://github.com/Bears-R-Us/arkouda/pull/4385))

- Update stream_benchmark (#3580, [PR #4562](https://github.com/Bears-R-Us/arkouda/pull/4562))
- Closes 4581 -- adds benchmark for where ([PR #4591](https://github.com/Bears-R-Us/arkouda/pull/4591))
- improve benchmark flags ([PR #4616](https://github.com/Bears-R-Us/arkouda/pull/4616))
- Update array_create_benchmark (#3561, [PR #4607](https://github.com/Bears-R-Us/arkouda/pull/4607))
- Update array_transfer_benchmark (#3562, [PR #4608](https://github.com/Bears-R-Us/arkouda/pull/4608))
- Update bigint_bitwise_binops_benchmark (#3563, [PR #4609](https://github.com/Bears-R-Us/arkouda/pull/4609))
- Update gather_benchmark (#3569, [PR #4612](https://github.com/Bears-R-Us/arkouda/pull/4612))
- Update scatter_benchmark (#3576, [PR #4615](https://github.com/Bears-R-Us/arkouda/pull/4615))
- Update dataframe_indexing_benchmark (#3566, [PR #4605](https://github.com/Bears-R-Us/arkouda/pull/4605))
- Fix a benchmark's output ([PR #4151](https://github.com/Bears-R-Us/arkouda/pull/4151))
- Refactored benchmarks: `stream`, `scatter`, `bitwise`, `array_create`, `gather`, etc. (#3580, #3561, #3563)
- Introduced `where` benchmark (#4581)





---


## Bug Fixes


- Read multiple Parquet row groups correctly ([PR #3989](https://github.com/Bears-R-Us/arkouda/pull/3989))
- fixes error in ak.trunc and ak.ceil, also adds unit tests (#3984, [PR #3985](https://github.com/Bears-R-Us/arkouda/pull/3985))
- flake8 errors in master (#4113, [PR #4114](https://github.com/Bears-R-Us/arkouda/pull/4114))
- Corrects error messages in conftest.py (#4117, [PR #4118](https://github.com/Bears-R-Us/arkouda/pull/4118))
- bug in standard_gamma when no seed is used (#4111, [PR #4115](https://github.com/Bears-R-Us/arkouda/pull/4115))
- Fix `xp.diff` ([PR #4128](https://github.com/Bears-R-Us/arkouda/pull/4128))
- Fix the shutdown test by using the correct server ([PR #4158](https://github.com/Bears-R-Us/arkouda/pull/4158))
- Fix can_cast with NumPy 2.0 breaking changes (#3337, [PR #4136](https://github.com/Bears-R-Us/arkouda/pull/4136))
- Closes-4173 by raising errors when using max_bits had been causing server crashes ([PR #4189](https://github.com/Bears-R-Us/arkouda/pull/4189))
- mypy errors in history.py (#4193, [PR #4196](https://github.com/Bears-R-Us/arkouda/pull/4196))
- Fix the .rst files (Segarray and Strings) after they were commented out during the Numpy Reorg (#4163, [PR #4191](https://github.com/Bears-R-Us/arkouda/pull/4191))
- Fix delete in arkouda/pdarraymanipulation.py (#4119, [PR #4192](https://github.com/Bears-R-Us/arkouda/pull/4192))
- Fixes flake8 errors in client.py ([PR #4215](https://github.com/Bears-R-Us/arkouda/pull/4215))
- Part of #4228:  CI failing due to dependency errors ([PR #4229](https://github.com/Bears-R-Us/arkouda/pull/4229))
- Fix max_bits on multidimensional arrays (#4203, [PR #4204](https://github.com/Bears-R-Us/arkouda/pull/4204))
- Fixes abs function in array_api/array_object ([PR #4226](https://github.com/Bears-R-Us/arkouda/pull/4226))
- Fix arkouda commit in containers following upgrade to arrow 19.0.1 ([PR #4359](https://github.com/Bears-R-Us/arkouda/pull/4359))
- Fix CSV reading failures due to array misuse ([PR #4384](https://github.com/Bears-R-Us/arkouda/pull/4384))
- Fixes bug involving arange and kwargs -- Closes 4321 ([PR #4379](https://github.com/Bears-R-Us/arkouda/pull/4379))
- Closes-4397 corrects errors in unit tests for PR 4361 ([PR #4398](https://github.com/Bears-R-Us/arkouda/pull/4398))
- Indexing problem crashes server (#4420, [PR #4422](https://github.com/Bears-R-Us/arkouda/pull/4422))
- failing numeric_tests (#4430, [PR #4431](https://github.com/Bears-R-Us/arkouda/pull/4431))
- resolve DOC602 pydoclint errors (#4402, [PR #4404](https://github.com/Bears-R-Us/arkouda/pull/4404))
- mypy error in index.py (#4451, [PR #4452](https://github.com/Bears-R-Us/arkouda/pull/4452))
- Fixes ak.full behavior, Closes 4312 ([PR #4447](https://github.com/Bears-R-Us/arkouda/pull/4447))
- fix chplcheck errors in ApplyMsg.chpl (#4436, [PR #4437](https://github.com/Bears-R-Us/arkouda/pull/4437))
- mypy error in groupbyclass.py (#4465, [PR #4466](https://github.com/Bears-R-Us/arkouda/pull/4466))
- Fix host-related issues re start_arkouda_server() ([PR #4454](https://github.com/Bears-R-Us/arkouda/pull/4454))
- bug in accessor module for series.str and series.dt (#4524, [PR #4525](https://github.com/Bears-R-Us/arkouda/pull/4525))
- Fix darglint errors in arkouda/testing/_equivalence_asserters.py ([PR #4566](https://github.com/Bears-R-Us/arkouda/pull/4566))
- Fixes `pydoclint` error when using user conda environment ([PR #4604](https://github.com/Bears-R-Us/arkouda/pull/4604))
- Fix testing of auto checkpointing ([PR #4592](https://github.com/Bears-R-Us/arkouda/pull/4592))
- Errors in multidimensional take testing (#4623, [PR #4636](https://github.com/Bears-R-Us/arkouda/pull/4636))
- Adds an if to avoid the error in Bug 4594 ([PR #4614](https://github.com/Bears-R-Us/arkouda/pull/4614))
- fix a typo in pytest.ini causing groupby tests to be skipped ([PR #4655](https://github.com/Bears-R-Us/arkouda/pull/4655))
- Temporarily don't timeout pytest for debugging ([PR #4686](https://github.com/Bears-R-Us/arkouda/pull/4686))
- Fix generic actual bug in `registry/register_commands.py` ([PR #4087](https://github.com/Bears-R-Us/arkouda/pull/4087))
- Fixed Parquet read/write bug when locale count changes (#4076, [PR #4077](https://github.com/Bears-R-Us/arkouda/pull/4077))
- Adds unit test for fix in PR #3950 ([PR #3968](https://github.com/Bears-R-Us/arkouda/pull/3968)) (closes #3967)
- trapz instability (#4489, [PR #4645](https://github.com/Bears-R-Us/arkouda/pull/4645))
- Fixes pydoclint error when using user conda environment ([PR #4604](https://github.com/Bears-R-Us/arkouda/pull/4604))

- Fixed permutation instability (#3974)
- Fixed `ak.ceil`, `ak.trunc`, `ak.permutation`, `ak.full` errors (#3984, #4312)
- Resolved segmented search crash ([PR #4367](https://github.com/Bears-R-Us/arkouda/pull/4367))
- Resolved issues with `max_bits`, `searchsorted`, and `repartition` functions (#4173, #4203, #4235)
- gh-pages action seems to need rsync (#4469, [PR #4470](https://github.com/Bears-R-Us/arkouda/pull/4470))
- Converting between multidimensional numpy and pdarrays of bigints (#4167, [PR #4168](https://github.com/Bears-R-Us/arkouda/pull/4168))
- Resolve server crashes in SegStringSort ([PR #4327](https://github.com/Bears-R-Us/arkouda/pull/4327))
- searchsorted to work with uint type (#4235, [PR #4440](https://github.com/Bears-R-Us/arkouda/pull/4440))

---






---




## Testing Improvements
- `skip_by_rank` enhanced for set containment (#3954, [PR #3955](https://github.com/Bears-R-Us/arkouda/pull/3955))
- Increases multidimensional testing for pdarraycreation module (#4028, [PR #4108](https://github.com/Bears-R-Us/arkouda/pull/4108))
- add `ARKOUDA_DEFAULT_TEMP_DIRECTORY` to globally set the default temp directory ([PR #4198](https://github.com/Bears-R-Us/arkouda/pull/4198))
- Detect server exit while reading connection file ([PR #4403](https://github.com/Bears-R-Us/arkouda/pull/4403))
- Improved launch scripts for PBS environments ([PR #4088](https://github.com/Bears-R-Us/arkouda/pull/4088))
- Avoid 0 in test with unsigned numbers ([PR #4104](https://github.com/Bears-R-Us/arkouda/pull/4104))
- Temporarily skip scipy testing for Python 3.13+ ([PR #4123](https://github.com/Bears-R-Us/arkouda/pull/4123))
- pytest marker for python version (#4130, [PR #4132](https://github.com/Bears-R-Us/arkouda/pull/4132))
- update unit tests handling of bigint (#4137, [PR #4138](https://github.com/Bears-R-Us/arkouda/pull/4138))
- Remove verbose output from comm diag test ([PR #4179](https://github.com/Bears-R-Us/arkouda/pull/4179))
- Add `pytest.temp_directory` ([PR #4181](https://github.com/Bears-R-Us/arkouda/pull/4181))
- Closes-4045 adds multi-dim tests to scipy_test ([PR #4172](https://github.com/Bears-R-Us/arkouda/pull/4172))
- Restore scipy skipifs removed in #4172 ([PR #4197](https://github.com/Bears-R-Us/arkouda/pull/4197))
- union1d needs more rigorous unit tests on Strings (#4317, [PR #4322](https://github.com/Bears-R-Us/arkouda/pull/4322))
- pdarray_creation_test.py unit test improvements Part 1 ([PR #4369](https://github.com/Bears-R-Us/arkouda/pull/4369))
- Closes-4567 aligns use of pytest.seed in numeric_test.py ([PR #4568](https://github.com/Bears-R-Us/arkouda/pull/4568))
- add pytest-timeout (#4445, [PR #4654](https://github.com/Bears-R-Us/arkouda/pull/4654))
- Increase unit test timeout to 30 minutes ([PR #4666](https://github.com/Bears-R-Us/arkouda/pull/4666))
- add logging to pytest (#4667, [PR #4669](https://github.com/Bears-R-Us/arkouda/pull/4669))
- Skipif test_checkpoint() ([PR #4690](https://github.com/Bears-R-Us/arkouda/pull/4690))


---


## Documentation



- docstring bug in standard_gamma (#4110, [PR #4112](https://github.com/Bears-R-Us/arkouda/pull/4112))
- Test examples in arkouda.util.py (#4004, [PR #4100](https://github.com/Bears-R-Us/arkouda/pull/4100))
- update docstrings arkouda.dtypes (#3535, [PR #4134](https://github.com/Bears-R-Us/arkouda/pull/4134))
- Closes 3547 updating docstrings in pdarrayclass.py ([PR #4109](https://github.com/Bears-R-Us/arkouda/pull/4109))
- Closes-3549 Updates docstrings in pdarraysetops ([PR #4171](https://github.com/Bears-R-Us/arkouda/pull/4171))
- Test examples in arkouda.client.py (#3994, [PR #4212](https://github.com/Bears-R-Us/arkouda/pull/4212))
- Closes-4195 docstrings in array_object.py and makes some minor executable changes. ([PR #4201](https://github.com/Bears-R-Us/arkouda/pull/4201))
- Test examples in arkouda.numpy._utils.py (#4003, [PR #4221](https://github.com/Bears-R-Us/arkouda/pull/4221))
- Test examples in arkouda.series.py (#4002, [PR #4222](https://github.com/Bears-R-Us/arkouda/pull/4222))
- add doctest to the project (#4237, [PR #4239](https://github.com/Bears-R-Us/arkouda/pull/4239))
- Update Docstrings arkouda.util (#3558, [PR #4307](https://github.com/Bears-R-Us/arkouda/pull/4307))
- Prevent comments in docstring Examples (#4365, [PR #4366](https://github.com/Bears-R-Us/arkouda/pull/4366))
- Adds doctest to a subset of modules ([PR #4370](https://github.com/Bears-R-Us/arkouda/pull/4370))
- Remove the DAR203 errors from the docstrings (#4355, [PR #4371](https://github.com/Bears-R-Us/arkouda/pull/4371))
- add doctest to strings module (#4282, [PR #4409](https://github.com/Bears-R-Us/arkouda/pull/4409))
- add doctest to pdarraysetops module (#4277, [PR #4413](https://github.com/Bears-R-Us/arkouda/pull/4413))
- add doctest to segarray module (#4280, [PR #4412](https://github.com/Bears-R-Us/arkouda/pull/4412))
- add doctest to sparsematrix module (#4288, [PR #4411](https://github.com/Bears-R-Us/arkouda/pull/4411))
- add doctest to pdarrayclass module (#4274, [PR #4414](https://github.com/Bears-R-Us/arkouda/pull/4414))
- add doctest to numeric module (#4273, [PR #4416](https://github.com/Bears-R-Us/arkouda/pull/4416))
- add doctest to plotting module (#4267, [PR #4415](https://github.com/Bears-R-Us/arkouda/pull/4415))
- add doctest to _stats_py module (#4291, [PR #4410](https://github.com/Bears-R-Us/arkouda/pull/4410))
- Resolve D205 docstring errors (#4340, [PR #4401](https://github.com/Bears-R-Us/arkouda/pull/4401))
- add doctest to alignment module (#4248, [PR #4435](https://github.com/Bears-R-Us/arkouda/pull/4435))
- skip strings module doctest when CHPL_COMM=ugni ([PR #4446](https://github.com/Bears-R-Us/arkouda/pull/4446))
- skips string module doctest unit test on apollo ([PR #4449](https://github.com/Bears-R-Us/arkouda/pull/4449))
- add doctest to apply module (#4249, [PR #4438](https://github.com/Bears-R-Us/arkouda/pull/4438))
- Closes 4423:  update documentation build ([PR #4424](https://github.com/Bears-R-Us/arkouda/pull/4424))
- add doctest to categorical module (#4250, [PR #4439](https://github.com/Bears-R-Us/arkouda/pull/4439))
- skip doctest unit tests for string, categorial on hpe systems ([PR #4456](https://github.com/Bears-R-Us/arkouda/pull/4456))
- Adds examples where they were missing from docstrings in numpy/numeric.py ([PR #4463](https://github.com/Bears-R-Us/arkouda/pull/4463))
- Add doctest to accessor module (#4247, [PR #4473](https://github.com/Bears-R-Us/arkouda/pull/4473))
- add doctest to array_api/manipulation_functions module (#4300, [PR #4508](https://github.com/Bears-R-Us/arkouda/pull/4508))
- add doctest to manipulation_functions module (#4272, [PR #4485](https://github.com/Bears-R-Us/arkouda/pull/4485))
- Resolve D400 docstring errors (#4339, [PR #4517](https://github.com/Bears-R-Us/arkouda/pull/4517))
- 4305 add doctest to array api typing module ([PR #4516](https://github.com/Bears-R-Us/arkouda/pull/4516))
- add doctest to array_api/creation_functions module (#4294, [PR #4513](https://github.com/Bears-R-Us/arkouda/pull/4513))
- add doctest to array_api/elementwise_functions module (#4297, [PR #4512](https://github.com/Bears-R-Us/arkouda/pull/4512))
- add doctest to array_api/searching_functions module (#4301, [PR #4507](https://github.com/Bears-R-Us/arkouda/pull/4507))
- add doctest to array_api/set_functions module (#4302, [PR #4509](https://github.com/Bears-R-Us/arkouda/pull/4509))
- add doctest to array_api/linalg module (#4299, [PR #4510](https://github.com/Bears-R-Us/arkouda/pull/4510))
- add doctest to array_api/indexing_functions module (#4298, [PR #4511](https://github.com/Bears-R-Us/arkouda/pull/4511))
- add doctest to array_api/sorting_functions module (#4303, [PR #4492](https://github.com/Bears-R-Us/arkouda/pull/4492))
- add doctest to array_api/statistical_functions module (#4304, [PR #4491](https://github.com/Bears-R-Us/arkouda/pull/4491))
- add doctest to groupbyclass module (#4255, [PR #4487](https://github.com/Bears-R-Us/arkouda/pull/4487))
- add doctest to array api constants module (#4293, [PR #4514](https://github.com/Bears-R-Us/arkouda/pull/4514))
- Remove the DAR202 errors from the docstrings (#4350, [PR #4558](https://github.com/Bears-R-Us/arkouda/pull/4558))
- docstring for infoclass module (#4548, [PR #4550](https://github.com/Bears-R-Us/arkouda/pull/4550))
- docstring for setup.py (#4528, [PR #4529](https://github.com/Bears-R-Us/arkouda/pull/4529))
- Remove the DAR102 errors from the docstrings (#4352, [PR #4553](https://github.com/Bears-R-Us/arkouda/pull/4553))
- add ellipses to floats in numeric.py docstrings to avoid precision related errors ([PR #4573](https://github.com/Bears-R-Us/arkouda/pull/4573))
- apply module docstring (#4576, [PR #4578](https://github.com/Bears-R-Us/arkouda/pull/4578))
- Closes 4579:  message module docstring ([PR #4580](https://github.com/Bears-R-Us/arkouda/pull/4580))
- add docstring for the match module (#4588, [PR #4589](https://github.com/Bears-R-Us/arkouda/pull/4589))
- add docstring for the alignment module (#4586, [PR #4587](https://github.com/Bears-R-Us/arkouda/pull/4587))
- docstring for categorical module (#4536, [PR #4537](https://github.com/Bears-R-Us/arkouda/pull/4537))
- docstring for logger module (#4544, [PR #4545](https://github.com/Bears-R-Us/arkouda/pull/4545))
- docstring for io_util (#4534, [PR #4535](https://github.com/Bears-R-Us/arkouda/pull/4535))
- add doctest to array_api/array_object module (#4292, [PR #4506](https://github.com/Bears-R-Us/arkouda/pull/4506))
- 4295 add doctest to array api data type functions module ([PR #4515](https://github.com/Bears-R-Us/arkouda/pull/4515))
- docstring for io module (#4582, [PR #4583](https://github.com/Bears-R-Us/arkouda/pull/4583))
- docstring for the security module (#4584, [PR #4585](https://github.com/Bears-R-Us/arkouda/pull/4585))
- adds missing docstrings to installers.py ([PR #4624](https://github.com/Bears-R-Us/arkouda/pull/4624))
- plotting module docstring improvement ([PR #4625](https://github.com/Bears-R-Us/arkouda/pull/4625))
- add docstring for client_dtypes (#4590, [PR #4595](https://github.com/Bears-R-Us/arkouda/pull/4595))
- adds missing docstrings to the message module ([PR #4641](https://github.com/Bears-R-Us/arkouda/pull/4641))
- add docstring for comm diagnostics module (#4601, [PR #4602](https://github.com/Bears-R-Us/arkouda/pull/4602))
- docstring for the history module (#4600, [PR #4603](https://github.com/Bears-R-Us/arkouda/pull/4603))
- adds missing docstrings for accessor module ([PR #4618](https://github.com/Bears-R-Us/arkouda/pull/4618))
- add skips for hpe systems to test_comm_diagnostics_docstrings ([PR #4658](https://github.com/Bears-R-Us/arkouda/pull/4658))
- add skip_doctest option to tests (#4659, [PR #4660](https://github.com/Bears-R-Us/arkouda/pull/4660))
- Test examples in `arkouda.sorting.py` (#4006, [PR #4084](https://github.com/Bears-R-Us/arkouda/pull/4084))
- Test examples in `arkouda.strings.py` (#4005, [PR #4090](https://github.com/Bears-R-Us/arkouda/pull/4090))
- Test examples in `arkouda.categorical.py` (#3993, [PR #4008](https://github.com/Bears-R-Us/arkouda/pull/4008))
- Test examples in `arkouda.alignment.py` (#3992, [PR #4007](https://github.com/Bears-R-Us/arkouda/pull/4007))
- Update docstrings in `arkouda.numeric` (#3546, [PR #3988](https://github.com/Bears-R-Us/arkouda/pull/3988))
- Update and correct docstrings in `pdarraycreation.py` (#3548, [PR #3987](https://github.com/Bears-R-Us/arkouda/pull/3987))
- numpy imports not showing up in docs (#4377, [PR #4380](https://github.com/Bears-R-Us/arkouda/pull/4380))


- Doctest coverage added across `strings`, `categorical`, `numeric`, `groupbyclass`, `array_api`, and more (#4282, #4274, #4291)
- Removed pydocstyle and darglint errors: `DAR102`, `DAR202`, `D400`, `D205`, `DOC602`, etc. (#4352, #4355, #4339)
- Numpy-style modules reorganized into dedicated directories (`numpy`, `pandas`, `scipy`) ([PR #4183](https://github.com/Bears-R-Us/arkouda/pull/4183))



---

## Developer Experience

- Runs isort and black on project ([PR #4139](https://github.com/Bears-R-Us/arkouda/pull/4139))
- update CI to use chapel 2.3.0 (#4085, [PR #4086](https://github.com/Bears-R-Us/arkouda/pull/4086))
- add flake8 checks for unit tests to CI (#3980, [PR #3983](https://github.com/Bears-R-Us/arkouda/pull/3983))
- Remove pinned Python versions and dependencies ([PR #4097](https://github.com/Bears-R-Us/arkouda/pull/4097))
- update ubuntu version in CI (#4160, [PR #4161](https://github.com/Bears-R-Us/arkouda/pull/4161))
- simplify makefile test in CI (#4205, [PR #4207](https://github.com/Bears-R-Us/arkouda/pull/4207))
- Create ubuntu docker container to use with the CI. (#4217, [PR #4218](https://github.com/Bears-R-Us/arkouda/pull/4218))
- add git pre-commit hooks #4356 (#4356, [PR #4357](https://github.com/Bears-R-Us/arkouda/pull/4357))
- add check mypy version to CI ([PR #4457](https://github.com/Bears-R-Us/arkouda/pull/4457))
- add cancel-in-progress to CI (#4698, [PR #4699](https://github.com/Bears-R-Us/arkouda/pull/4699))


- Add memory logging to the CI ([PR #4695](https://github.com/Bears-R-Us/arkouda/pull/4695))
- Add external tools documentation (#4079, [PR #4080](https://github.com/Bears-R-Us/arkouda/pull/4080))
- Add retry on installation steps in CI (#3978, [PR #3979](https://github.com/Bears-R-Us/arkouda/pull/3979))
- Remove duplicate Dockerfile (#3976, [PR #3977](https://github.com/Bears-R-Us/arkouda/pull/3977))
- Simplify offline builds (#3957, #3944, [PR #3958](https://github.com/Bears-R-Us/arkouda/pull/3958))
- Removed duplicate Dockerfile (#3976)

- CI updated for Python 3.13, Chapel 2.3, and better dependency reuse (#4085, #4228)
- Added: `flake8`, `ruff`, `darglint`, `pydocstyle`, `pydoclint`, `docstr-coverage`, `chplcheck`, `pytest-timeout` (#3980, #4113, #4402, #4386, #4388, #4445)
- Git pre-commit hooks added (#4356)

- remove type-ignore from arange (#4395, [PR #4396](https://github.com/Bears-R-Us/arkouda/pull/4396))
- arkouda_benchmark_linux and arkouda_tests_linux to use ubuntu-with-arkouda-deps (#4236, [PR #4238](https://github.com/Bears-R-Us/arkouda/pull/4238))


**Linters**
- add pydocstyle to the project (#4320, [PR #4324](https://github.com/Bears-R-Us/arkouda/pull/4324))
- add pydoclint to the project (#4363, [PR #4364](https://github.com/Bears-R-Us/arkouda/pull/4364))
- add docstr-coverage (#4386, [PR #4387](https://github.com/Bears-R-Us/arkouda/pull/4387))
- add chplcheck (#4388, [PR #4389](https://github.com/Bears-R-Us/arkouda/pull/4389))
- docstr-coverage ignore for overload functions ([PR #4432](https://github.com/Bears-R-Us/arkouda/pull/4432))
- add darglint to the project (#4343, [PR #4345](https://github.com/Bears-R-Us/arkouda/pull/4345))

**Makefile Improvements**
- make doc-clean should depend on stub-clean (#4155, [PR #4156](https://github.com/Bears-R-Us/arkouda/pull/4156))
- add isort to Makefile (#4199, [PR #4211](https://github.com/Bears-R-Us/arkouda/pull/4211))
- Closes 4206:  add install-pytables to Makefile ([PR #4208](https://github.com/Bears-R-Us/arkouda/pull/4208))
- add ruff-format to Makefile (#4194, [PR #4200](https://github.com/Bears-R-Us/arkouda/pull/4200))
- Add make format (#4231, [PR #4232](https://github.com/Bears-R-Us/arkouda/pull/4232))
- Makefile additions: `make format`, `make ruff-format`, `make isort`, `install-pytables` (#4206, #4211, #4200)

---



## New Contributors



- [@1RyanK](https://github.com/1RyanK) – multiple core features, DataFrame fixes, testing infrastructure, and doc coverage
- [@alvaradoo](https://github.com/alvaradoo) – environment improvements and CI



---



## Full Changelog



[Compare full changes from v2025.01.13 to this release](https://github.com/Bears-R-Us/arkouda/compare/v2025.01.13...v2025)