# EES-Tokenizer

Turn a pooled plant single-cell transcriptome into a set of discrete gene tokens,
against a **fixed per-gene reference distribution** that ships with this
repository.

A gene whose pooled value sits above the 95th percentile of its own reference
distribution emits `GENE_UP`. A gene with a **non-zero** value below its 5th
emits `GENE_DOWN`. Everything else emits nothing. Magnitude is discarded.

```bash
pip install -e .
ees-tokenize --h5ad my_data.h5ad \
             --reference reference/gene_percentiles_pb.csv \
             --vocab reference/ees_vocab.txt \
             --label_col cluster --out_prefix my_tokens
```

No trained model is needed to produce tokens, and there is no deep-learning
dependency. The model that was trained on these tokens lives in
[ees-transformer](https://github.com/k821209/ees-transformer); the corpus and
weights are at [10.5281/zenodo.22741864](https://doi.org/10.5281/zenodo.22741864).

## Why the reference is fixed

An UP call is meaningful only relative to some distribution. A tokenizer that
recomputes percentiles on each new dataset emits tokens that mean something
different from the ones anything was trained on: the same gene at the same
expression level would be UP in one dataset and silent in another purely because
the surrounding samples differ. So the percentiles are estimated once, over
181,660 pseudo-bulks of 363,361 *Arabidopsis* cells from ten scPlantDB datasets,
and then applied unchanged — the way a language model's vocabulary is fixed once
training ends.

**`reference/gene_percentiles_pb.csv`** has one row per gene:

| column | meaning |
| --- | --- |
| `q_lower`, `q_upper` | the rule. 36,534 of 53,678 genes carry them; the rest were detected in fewer than 50 pseudo-bulks and emit nothing |
| `expressing_pbs` | how many pseudo-bulks the gene was detected in — the support behind its threshold |
| `mean_expr_nonzero` | mean of its non-zero pooled values, descriptive |

All 53,678 rows matter even though only 36,534 can emit tokens: the full set is
the **gene universe the library size is summed over**. See *Getting the scale
right* below.

## Getting the scale right

The thresholds carry units. They are in

```
log1p( count / library_size × 10⁴ )
```

over a pool of ten cells summed from **raw integer counts**. Three things have to
match or the numbers stop meaning anything while the output still looks fine:

1. **Raw counts in, not normalised values.** If your matrix is already
   log-normalised, pooling it applies the transform twice.
2. **Join genes by identifier, never by position.** Your gene axis differs from
   ours in order and in length. A positional join silently attaches one gene's
   threshold to another and produces output that parses correctly and names the
   wrong genes.
3. **The library size is summed over the reference gene universe**, all 53,678
   rows of the table, not over whatever your object happens to carry. Genes we
   know and you lack contribute zero either way; genes *you* carry and we never
   saw would otherwise inflate your denominator and shift every value off the
   scale the thresholds live on. This is the default. Every run reports
   `count_fraction_on_reference_genes`, and a value well below 1 means your
   denominator is being set by material the reference never saw.

Every run also reports the three coverage counts, because each one quietly
shrinks what a model downstream can see: genes here with no threshold, reference
genes absent here, and emitted tokens outside the vocabulary (`[UNK]`).

## Already-bulk input, and why the bulk table is a different artifact

If your input is already one measurement per sample — bulk RNA-seq, or anything
pre-aggregated — pooling would average away the sample you care about. Use
`--no_pooling`, which emits one sentence per sample and emits **all** of them:

```bash
ees-tokenize --h5ad my_bulk.h5ad --no_pooling \
             --reference <a reference built for that data> \
             --down_rule with_zeros --out_prefix my_tokens
```

Two things have to change together, and only one of them is a flag.

**The rule.** `DOWN` here requires `0 < x < q_lower`, so a feature at zero emits
nothing: absence and low expression are deliberately different symbols. That is
correct for the pooled reference shipped in this repository, whose percentiles
were taken over each gene's non-zero pooled values. A reference whose percentiles
were taken over **all** samples including zeros — as the bulk arm of the paper
did — means something else by DOWN, and needs `--down_rule with_zeros`. Passing a
table built one way with the rule of the other produces well-formed tokens that
mean something different, and nothing raises an error. Every run records which
rule it used in the coverage report.

**The table.** The reference in `reference/` is **not usable for bulk**, and not
because of the rule. It is gene-level (`AT2G01170`) and estimated on pooled
single-cell data; the paper's bulk arm quantifies transcripts (`AT2G01170.1`)
over 30,647 features with its own thresholds and its own 46,728-token
vocabulary. The two vocabularies are not interchangeable and a model trained on
one cannot read sentences written in the other. The bulk tables live in
[ees-transformer](https://github.com/k821209/ees-transformer). If you point this
tool at bulk data with the table shipped here, the identifier join will match
almost nothing and the run will say so.

## Reproducibility

`tests/test_roundtrip.py` tokenizes pseudo-bulks of the published corpus with
this table and checks the token sets against the deposited sentences. Applied to
the corpus it reproduces them exactly.

Applied **one study at a time**, with nothing else loaded, it reproduces that
study's corpus sentences for **98.95%** of the deepest source study's 10,095
pools and **99.49%** of the shallowest study's 5,098. Every mismatch is a single
token whose value falls on a threshold, i.e. floating-point rounding across the
comparison. Treat ~99% as the expected same-input reproduction rate: the table is
portable one dataset at a time, which is the point of releasing it.

## Three bounds to read with the table

**The thresholds are unequally determined.** `q_upper` is pinned by the top 5% of
a gene's expressing pseudo-bulks — 2.8 observations at the first percentile of
genes, 8,495 at the ninety-fifth, and 19.5% of thresholded genes rest on fewer
than 20. Those weak ones fire rarely and carry 0.07% of emitted UP tokens, so a
large corpus is barely affected, but if you tokenize a small dataset, check
`expressing_pbs` for the genes your conclusion rests on.

**Fewer independent observations than pools.** Each cell entered about five
pseudo-bulks, so a threshold rests on 363,361 cells counted roughly five times
over rather than on 181,660 independent draws.

**The equinumerous guarantee is global, not per class.** Over the whole corpus
UP : DOWN is 1.00 by construction (90,433,368 : 90,433,344). Within a single cell
type it runs from 0.14 to 10.16 — a factor of 74 — and across the ten source
files from 0.17 to 3.43. This is not a small-sample effect: the deviation is
uncorrelated with how much of the corpus a cell type contributes. A token is
defined against the whole corpus, so a cell type's token profile depends on what
else that corpus holds, and this one is majority root. **Do not read your own
UP : DOWN ratio as a correctness check.** A ratio far from 1 is the expected
result for a narrow dataset, not a bug.

Two further things worth knowing about DOWN specifically. `DOWN` requires
`0 < x < q_lower`, so it gates on *detection*: a gene at zero emits nothing at
all. Absence and low expression are deliberately different symbols, and the
consequence is that DOWN counts respond to sequencing depth in a way UP counts
do not.

## Extending the reference

What the table needs is breadth of composition, not more cells. Adding more of
what already dominates the corpus changes nothing. If you are assembling data
toward a future version, the axes that matter are which cell types are
represented and over what range of sequencing depth.

## Citation

See `CITATION.cff`. Please cite the paper and the Zenodo deposit.
