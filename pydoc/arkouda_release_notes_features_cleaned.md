## New Features

**Array Functions**

- tile function (#3003, [PR #4101](https://github.com/Bears-R-Us/arkouda/pull/4101))
- Add `xp.trapz` ([PR #4127](https://github.com/Bears-R-Us/arkouda/pull/4127))
- repeat function (#3000, [PR #4146](https://github.com/Bears-R-Us/arkouda/pull/4146))
- nextafter function (#3004, [PR #4219](https://github.com/Bears-R-Us/arkouda/pull/4219))
- Closes-4392 adds ak.newaxis constant ([PR #4393](https://github.com/Bears-R-Us/arkouda/pull/4393))
- reshape function and max_bits handling (#4165, [PR #4394](https://github.com/Bears-R-Us/arkouda/pull/4394))
- Adds negative axis handling to squeeze ([PR #4406](https://github.com/Bears-R-Us/arkouda/pull/4406))
- diff function (#2998, [PR #4418](https://github.com/Bears-R-Us/arkouda/pull/4418))
- take function (#3755, [PR #4419](https://github.com/Bears-R-Us/arkouda/pull/4419))
- Closes 4425 (adds axis handling to mean, var, std) ([PR #4442](https://github.com/Bears-R-Us/arkouda/pull/4442))
- trapz instability (#4489, [PR #4645](https://github.com/Bears-R-Us/arkouda/pull/4645))
- Added `ak.tile`, `ak.repeat`, `ak.take`, `ak.diff`, `ak.nextafter`, `ak.newaxis`, `ak.quantile`, `ak.percentile`, `ak.vecdot`, `ak.result_type`, `ak.eye`, and `ak.append` (#3000, #2998, #3004, #3292, #4483, #4502)

**Checkpointing and Logging**

- Use server's mechanism to redirect logs to a file ([PR #4152](https://github.com/Bears-R-Us/arkouda/pull/4152))
- Part of #2384: Auto checkpoint upon exceeding memory percentage or idle time ([PR #4391](https://github.com/Bears-R-Us/arkouda/pull/4391))
- Fewer 'throws' while logging ([PR #4433](https://github.com/Bears-R-Us/arkouda/pull/4433))
- Simplify and extend logic in binopvv (#4459, [PR #4462](https://github.com/Bears-R-Us/arkouda/pull/4462))
- Part of #2384: Preserve previous auto-checkpoint ([PR #4549](https://github.com/Bears-R-Us/arkouda/pull/4549))
- Simplify and extend logic in binopvs (#4460, [PR #4563](https://github.com/Bears-R-Us/arkouda/pull/4563))
- Initial capability to checkpoint partial server state ([PR #3915](https://github.com/Bears-R-Us/arkouda/pull/3915))
- Experimental checkpointing of server state (#2384)

**Deprecations and Refactors**

- remove deprecated lookup function (#4375, [PR #4376](https://github.com/Bears-R-Us/arkouda/pull/4376))
- refactor arange to use instantiateAndRegister (#4382, [PR #4383](https://github.com/Bears-R-Us/arkouda/pull/4383))

**Internal Improvements**

- parse_single_value to handle uints represented as nega… (#4147, [PR #4150](https://github.com/Bears-R-Us/arkouda/pull/4150))
- Change some nested foralls to for/foralls in HistogramMsg ([PR #4180](https://github.com/Bears-R-Us/arkouda/pull/4180))
- remove try! in toSymEntry (#4175, [PR #4176](https://github.com/Bears-R-Us/arkouda/pull/4176))
- Workaround an issue with using PrivateSpace domains in formals ([PR #4633](https://github.com/Bears-R-Us/arkouda/pull/4633))

**Other**

-  ([PR #4604](https://github.com/Bears-R-Us/arkouda/pull/4604))
- skip_by_rank to handle set containment (#3954, [PR #3955](https://github.com/Bears-R-Us/arkouda/pull/3955))
- Add Standard Gamma Distribution Function to the Random Module (Final Version) (#3846, [PR #4089](https://github.com/Bears-R-Us/arkouda/pull/4089))
- Closes 4028, addresses comments ([PR #4108](https://github.com/Bears-R-Us/arkouda/pull/4108))
- make doc-clean should depend on stub-clean (#4155, [PR #4156](https://github.com/Bears-R-Us/arkouda/pull/4156))
- Closes 4140, reverts to previous ak.zeros code ([PR #4141](https://github.com/Bears-R-Us/arkouda/pull/4141))
- add `ARKOUDA_DEFAULT_TEMP_DIRECTORY` to globally set the default temp directory ([PR #4198](https://github.com/Bears-R-Us/arkouda/pull/4198))
- Remove deprecated functions (#4308, [PR #4309](https://github.com/Bears-R-Us/arkouda/pull/4309))
- Removed deprecated functions ([PR #4374](https://github.com/Bears-R-Us/arkouda/pull/4374))
- Detect server exit while reading connection file ([PR #4403](https://github.com/Bears-R-Us/arkouda/pull/4403))
- append (#4502, [PR #4564](https://github.com/Bears-R-Us/arkouda/pull/4564))
- bool alias for bool_ (#4627, [PR #4628](https://github.com/Bears-R-Us/arkouda/pull/4628))
- Part of #2384: Checkpoint numeric arrays; improved framework ([PR #4644](https://github.com/Bears-R-Us/arkouda/pull/4644))
- Simplify offline builds (#3957, #3944, [PR #3958](https://github.com/Bears-R-Us/arkouda/pull/3958))
- `skip_by_rank` enhanced for set containment (#3954)
- Read multiple Parquet row groups correctly ([PR #3989](https://github.com/Bears-R-Us/arkouda/pull/3989))
- Removed duplicate Dockerfile (#3976)
- Upgraded to Apache Arrow 19.0.0 (#3981)
- Introduced `ak.apply` (#3963)
---

**Project Infrastructure**

- add pydocstyle to the project (#4320, [PR #4324](https://github.com/Bears-R-Us/arkouda/pull/4324))
- add pydoclint to the project (#4363, [PR #4364](https://github.com/Bears-R-Us/arkouda/pull/4364))
- add docstr-coverage (#4386, [PR #4387](https://github.com/Bears-R-Us/arkouda/pull/4387))
- add chplcheck (#4388, [PR #4389](https://github.com/Bears-R-Us/arkouda/pull/4389))
- docstr-coverage ignore for overload functions ([PR #4432](https://github.com/Bears-R-Us/arkouda/pull/4432))
- add missing __all__ (#4426, [PR #4427](https://github.com/Bears-R-Us/arkouda/pull/4427))
- gh-pages action seems to need rsync (#4469, [PR #4470](https://github.com/Bears-R-Us/arkouda/pull/4470))
- Tweaks a few files to avoid relying on CHPL_HOME ([PR #4551](https://github.com/Bears-R-Us/arkouda/pull/4551))

**String and Set Operations**

- Make an unordered set union of two Strings arrays function (#4244, [PR #4245](https://github.com/Bears-R-Us/arkouda/pull/4245))
- Create a repartitionByLocaleString function (#4497, [PR #4557](https://github.com/Bears-R-Us/arkouda/pull/4557))
- Create a repartitionByHashString function (#4499, [PR #4617](https://github.com/Bears-R-Us/arkouda/pull/4617))
- Server functions `repartitionByLocaleString` and `repartitionByHashString` added (#4497, #4499)

**Third-party Upgrades**

- upgrade to arrow 19.0.0 (#3981, [PR #3982](https://github.com/Bears-R-Us/arkouda/pull/3982))
- Pr/3831 ([PR #3986](https://github.com/Bears-R-Us/arkouda/pull/3986))
- Closes Ticket #4341:  Upgrade to arrow 19.0.1 ([PR #4342](https://github.com/Bears-R-Us/arkouda/pull/4342))