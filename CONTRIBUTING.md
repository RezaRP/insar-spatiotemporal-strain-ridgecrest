# Contributing

This repository accompanies a specific manuscript, so its primary obligation is that the
published numbers remain reproducible. Contributions are welcome within that constraint.

## Reporting a problem

Open an issue. The most useful reports include the command you ran, the Python and package
versions, and the number you got where the manuscript or `docs/methods.md` says something
different. A discrepancy in a published value is the highest-priority kind of issue here —
please say so explicitly in the title.

## Changing analysis code

Any change that could move a published number must state so in the pull request, and must
either (a) leave the published values unchanged, with the test suite proving it, or (b)
explain precisely which value changes and why the new one is correct.

Frozen quantities — do not change casually:

- reference epoch 2017-05-27 and the P463 spatial datum;
- the vertical GP length scale (15 km), support bounds (35–120 km), and the pre-event
  frozen support topology;
- the condition-number rejection threshold (8);
- the strain support radius (8 km), bandwidth (4 km), and minimum sample count (16);
- the cokriging Matérn-3/2 length scale (8 km) and 24-km donor support;
- the detection calibration boundary of 2019-05-29;
- the `k = 0.5` CUSUM reference parameter and the `|z| ≥ 1.96`, ≥4-cell cluster rules.

## Style

```bash
ruff check src tests scripts
pytest -q
```

Both must pass. Line length 100. Type annotations on public functions.

## Notebooks

Edit the jupytext `.py` twin, never the `.ipynb` directly. Commit notebooks with outputs
stripped:

```bash
jupytext --to notebook <name>.py
python scripts/populate_repo.py --help   # see strip_notebook()
```

A pull request that adds cell outputs to a committed `.ipynb` will be asked to strip them.

## Data

Never commit `.h5`, `.tif`, `.npz`, or `.tenv3` files. `.gitignore` blocks them; if you find
yourself using `git add -f`, the file belongs on Zenodo instead. Add its provenance to
`data/manifests/dataset_manifest.csv`.

## Citation metadata

If you change `CITATION.cff` or `.zenodo.json`, the metadata CI job validates them. Keep
the version fields in `CITATION.cff`, `.zenodo.json`, and `pyproject.toml` in sync.
