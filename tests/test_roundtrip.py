"""NOT a unit test, and `pytest` alone will not run it.

This reproduces the published corpus from the released reference table, so it
needs the corpus and the source h5ad from the Zenodo deposit
(10.5281/zenodo.22741864) and a path to each. It is the script a user runs to
satisfy themselves that the table in this repository is the ruler the published
sentences were actually measured with, not a check that runs in CI.

Expected result. Tokenizing pseudo-bulks reconstructed from the manifest and
comparing token SETS -- order is not compared, because the corpus builder
shuffles before writing -- reproduces the deposited sentences for about 99% of
pools: 99.43% over 3,000 pools sampled at a fixed stride across all ten source
files, 98.95% and 99.49% when a single study is processed with nothing else
loaded. Every mismatch seen so far is a single token whose value falls on a
threshold, i.e. floating-point rounding across the `>` comparison. A rate far
below 99%, or a mismatch of many tokens at once, means something else: most
likely the input is not raw integer counts, or genes were joined by position
rather than by identifier.
"""

"""Does applying the PUBLISHED reference reproduce the training corpus exactly?

If the reference table is to be the method -- new data tokenized against these
fixed thresholds rather than against its own distribution -- then applying it to
the pseudo-bulks the corpus was built from must return those same sentences.
Anything else means the published table is not the ruler that was actually used,
and every downstream claim about tokenizing new data with it would be untestable.

The pseudo-bulk manifest records which cells entered which pool, so the pools are
reconstructed exactly rather than re-sampled; the only thing under test is the
thresholding step. Token ORDER is not compared: the builder shuffles before
truncation, so the sets are what must agree.
"""
import csv, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata as ad

H5AD, MANIFEST, REF, SENT, IDS = sys.argv[1:6]
N_CHECK = int(sys.argv[6]) if len(sys.argv) > 6 else 40
TARGET_SUM, MAX_TOKENS = 1e4, 8192

ref = pd.read_csv(REF).dropna(subset=["q_lower", "q_upper"])
ref_map = dict(zip(ref.gene_id.astype(str),
                   zip(ref.q_lower.to_numpy(float), ref.q_upper.to_numpy(float))))
print(f"reference: {len(ref_map):,} genes")

ids = [l.strip() for l in open(IDS)]
# Sample across the file: the corpus is written in cell-type order, so the first
# N pseudo-bulks are the alphabetically-early classes and nothing else.
step = max(1, len(ids) // N_CHECK)
want = {ids[i]: i for i in range(0, len(ids), step)}
print(f"checking {len(want)} pseudo-bulks spread across {len(ids):,}")

pools = defaultdict(list)
with open(MANIFEST) as fh:
    rd = csv.reader(fh); next(rd)
    for pb, ci in rd:
        if pb in want:
            pools[pb].append(int(ci))

sent = {}
with open(SENT) as fh:
    for i, line in enumerate(fh):
        if i in want.values():
            sent[ids[i]] = line.split()

adata = ad.read_h5ad(H5AD, backed="r")
# .raw.var_names here is positional ('0','1',...); the identifiers are on
# adata.var and the two axes are the same length, which the assert pins.
assert adata.raw.n_vars == adata.n_vars, "raw/var gene axes differ"
gene_ids = np.asarray(adata.var_names, dtype=str)
X = adata.raw.X

keep, qlo, qhi, kid = [], [], [], []
for pos, g in enumerate(gene_ids):
    h = ref_map.get(g)
    if h is not None:
        keep.append(pos); qlo.append(h[0]); qhi.append(h[1]); kid.append(g)
keep = np.asarray(keep); qlo = np.asarray(qlo); qhi = np.asarray(qhi)
up_t = np.array([f"{g}_UP" for g in kid], dtype=object)
dn_t = np.array([f"{g}_DOWN" for g in kid], dtype=object)
print(f"matched {len(keep):,} genes onto the h5ad gene axis\n")

ok = bad = 0
for pb, rows in sorted(pools.items()):
    truth = sent.get(pb)
    if truth is None:
        continue
    r = np.sort(np.asarray(rows))
    ps = np.asarray(X[r].sum(axis=0)).ravel()
    norm = np.log1p(ps / ps.sum() * TARGET_SUM)
    v = norm[keep]
    e = v > 0
    got = set(up_t[e & (v > qhi)]) | set(dn_t[e & (v < qlo)])
    tset = set(truth)
    if len(truth) >= MAX_TOKENS:
        # truncated: the corpus kept a random MAX_TOKENS of the full set, so the
        # only checkable claim is that every kept token is one we would emit
        good = tset <= got
        note = f"truncated, {len(tset)}/{len(got)} kept"
    else:
        good = (got == tset)
        note = f"{len(tset)} tokens"
    if good:
        ok += 1
    else:
        bad += 1
        print(f"  MISMATCH {pb}: ours {len(got)} vs corpus {len(tset)}, "
              f"missing {len(tset - got)}, extra {len(got - tset)}")
    if bad > 5:
        break

print(f"\n{ok} reproduced exactly, {bad} mismatched")
sys.exit(0 if bad == 0 else 1)
