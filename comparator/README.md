# [Comparator](https://github.com/leanprover/comparator)

Follow the [upstream setup and sandbox requirements](https://github.com/leanprover/comparator/blob/19e111e2141cf333c7daff0f64c5f24acc91dd2e/README.md), with
`landrun`, `lean4export`, and `nanoda_bin` on `PATH`. From the repository root
inside that sandbox:

```sh
lake exe cache get
lake exe comparator comparator/main.json
```

The checks assume the [three documented project axioms](../README.md).
