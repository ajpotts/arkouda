# Arrow Dataset refactor build notes

## 1) Ensure Arrow was built with Dataset enabled

Check:

```bash
ls -1 dep/arrow-install/lib/libarrow_dataset* dep/arrow-install/lib/libarrow_acero*
```

If `libarrow_dataset` is missing, rebuild Arrow with at least:

- `-DARROW_DATASET=ON`
- `-DARROW_CSV=ON` (needed for CSV dataset)
- `-DARROW_FILESYSTEM=ON`
- `-DARROW_ACERO=ON` (often pulled in by dataset execution)

Example (append to the cmake line in `dep/arrow.mk` via `ARROW_OPTIONS`):

```bash
-DARROW_DATASET=ON -DARROW_CSV=ON -DARROW_FILESYSTEM=ON -DARROW_ACERO=ON
```

## 2) Link Arkouda against Arrow Dataset

Your current link line includes `-larrow` and `-lparquet`, but *not* the dataset library.
Add `-larrow_dataset` (and typically `-larrow_acero`) to the Arrow shim link flags.

In Arkouda this is usually in `make/prologue/arrow_shims.mk` (or whatever centralizes Arrow libs).
Look for the variable that contains the Arrow libraries (often something like `ARROW_LIBS`, `ARROW_LDLIBS`, or `LIBS +=`).

Add:

```make
ARROW_LIBS += -larrow_dataset -larrow_acero
```

If your link is sensitive to ordering (static libs), keep dataset **before** `-larrow`.

## 3) Chapel interop const correctness

`c_readAllCols` now takes `const int* types` to match Chapel passing a const pointer.
