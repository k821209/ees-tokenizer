"""Tokenize a dataset into EES sentences against a FIXED reference distribution.

WHY THIS EXISTS. An EES token is defined by a per-gene percentile: a value above
that gene's 95th percentile emits `gene_UP`, a non-zero value below its 5th
emits `gene_DOWN`. `build_pseudobulk_ees_v2.py` derives those percentiles from
whatever data it is given, which is correct when building the training corpus
and wrong for every use afterwards. A dataset tokenized against its own
distribution produces tokens that mean something different from the ones the
model was trained on -- the same gene at the same expression level would be UP
in one dataset and silent in another, purely because the surrounding samples
differ. The thresholds are therefore published as an artifact and applied as a
fixed ruler, exactly as a trained model's vocabulary is fixed.

This is also what makes the trained model usable on data it has never seen. It
is the deployment path, and it is what the leave-one-study-out evaluation
assumes.

GENE ALIGNMENT IS BY IDENTIFIER, NEVER BY POSITION. The corpus builder indexes
the reference positionally, which is safe only because it wrote the reference
from the same matrix moments earlier. A new dataset has a different gene axis --
different order, different length, genes the reference never saw and genes it
has that this dataset lacks. Aligning by position here would silently apply
AT1G01010's threshold to some unrelated gene and the output would look
perfectly well-formed. Every join below is on the gene identifier.

THREE COVERAGE FACTS THE OUTPUT REPORTS, because each one silently shrinks what
the model can see:

  genes in this dataset with no reference threshold   -> emit nothing
  genes in the reference absent from this dataset     -> simply never fire
  tokens outside the model vocabulary                 -> become [UNK]

The third is a property of the published pair: the reference carries thresholds
for 36,534 genes while the vocabulary covers 31,594, so 4,941 genes can emit a
token that the model cannot represent. In the training corpus those account for
0.035% of token instances, but a dataset enriched for them would fare worse, so
the number is measured per run rather than assumed.

NORMALIZATION MUST MATCH THE THRESHOLDS. The reference percentiles were computed
on log1p(pool_sum / library_size * 10,000) over raw counts. Applying them to
anything else -- CPM without the log, log2, already-normalized input -- compares
values to a ruler built for a different scale. The `--use_raw` default reads
`.raw.X` for exactly this reason: `.X` in the source h5ad holds raw counts in
only 2 of 10 files and log-normalized values in the other 8.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Emit EES sentences using fixed reference percentiles.")
    p.add_argument("--h5ad", required=True,
                   help="Dataset to tokenize.")
    p.add_argument("--reference", required=True,
                   help="gene_percentiles_pb.csv: gene_id,q_lower,q_upper,...")
    p.add_argument("--vocab", default=None,
                   help="ees_vocab.txt. Optional; only used to report which "
                        "emitted tokens the model would read as [UNK].")
    p.add_argument("--metadata", default=None,
                   help="CSV mapping cell barcode -> cell type (and optionally "
                        "source study). Without it every cell forms one group.")
    p.add_argument("--cell_col", default="cell_id")
    p.add_argument("--label_col", default="cell_type")
    p.add_argument("--study_col", default=None,
                   help="Pool only within a study as well as within a label.")
    p.add_argument("--out_prefix", required=True,
                   help="Writes {prefix}_sentences.txt, _cell_ids.txt, "
                        "_metadata.csv, _coverage.json")
    p.add_argument("--k", type=int, default=10,
                   help="Cells per pseudo-bulk. 1 tokenizes cells directly, "
                        "which the thresholds were NOT built for (see --help "
                        "note in the docstring).")
    p.add_argument("--n_pools_div", type=int, default=2)
    p.add_argument("--target_sum", type=float, default=1e4)
    p.add_argument("--lib_over_dataset_axis", action="store_true",
                   help="sum the library size over this dataset's whole gene "
                        "axis instead of the reference universe. Only for "
                        "reproducing output from before that was made explicit.")
    p.add_argument("--no_pooling", action="store_true",
                   help="one sentence per sample, no pooling. Equivalent to "
                        "--k 1. Use it when the input is already a bulk or "
                        "otherwise pre-aggregated measurement, where pooling "
                        "would average away the sample you care about. Every "
                        "sample is emitted, not half of them.")
    p.add_argument("--down_rule", choices=("nonzero", "with_zeros"),
                   default="nonzero",
                   help="what DOWN means, and it MUST match the table you pass. "
                        "'nonzero' (default, the pooled single-cell reference "
                        "shipped here): DOWN requires 0 < x < q_lower, so a "
                        "feature at zero emits nothing and absence is not the "
                        "same symbol as low. 'with_zeros' (a reference whose "
                        "percentiles were taken over all samples INCLUDING "
                        "zeros, as the bulk arm of the paper did): a zero "
                        "counts as DOWN. Mixing a table built one way with the "
                        "rule of the other produces well-formed tokens that "
                        "mean something else, with no error raised.")
    p.add_argument("--min_tokens", type=int, default=10)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_raw", action="store_true", default=True,
                   help="Read .raw.X (counts). On by default: the thresholds "
                        "assume counts.")
    p.add_argument("--no_use_raw", dest="use_raw", action="store_false")
    return p.parse_args()


def _resolve_pooling(a):
    """--no_pooling is a readable spelling of k = 1; they must not disagree."""
    if a.no_pooling:
        if a.k not in (1, 10):
            sys.exit(f"--no_pooling with --k {a.k} is contradictory; drop one")
        a.k = 1
    return a


def load_reference(path):
    """gene_id -> (q_lower, q_upper), keeping only genes with both."""
    import pandas as pd

    df = pd.read_csv(path)
    need = {"gene_id", "q_lower", "q_upper"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"reference is missing column(s): {sorted(missing)}")
    df = df.dropna(subset=["q_lower", "q_upper"])
    if df.empty:
        sys.exit("reference has no gene with both q_lower and q_upper")
    if df["gene_id"].duplicated().any():
        dup = df.loc[df["gene_id"].duplicated(), "gene_id"].iloc[0]
        sys.exit(f"reference has duplicate gene_id, e.g. {dup!r}")
    return dict(zip(df["gene_id"].astype(str),
                    zip(df["q_lower"].to_numpy(np.float64),
                        df["q_upper"].to_numpy(np.float64))))


def load_universe(path):
    """Every gene_id the reference table lists, thresholded or not.

    This is the denominator's gene set and it is NOT the same as the set that
    can emit tokens. The corpus the thresholds were estimated on summed each
    pool's library size over its whole 53,678-gene axis, which is exactly the
    rows of this table; only 36,534 of them cleared the 50-pseudo-bulk floor and
    carry (q_lower, q_upper). Summing the denominator over the thresholded
    subset instead would put every value on a different scale from the one the
    thresholds live on.
    """
    import pandas as pd

    ids = pd.read_csv(path, usecols=["gene_id"])["gene_id"].astype(str)
    if ids.duplicated().any():
        sys.exit("reference has duplicate gene_id")
    return set(ids)


def main():
    a = _resolve_pooling(parse_args())
    import anndata as ad
    import pandas as pd

    ref = load_reference(a.reference)
    universe = load_universe(a.reference)
    print(f"reference: {len(ref):,} genes carry a threshold pair", flush=True)

    adata = ad.read_h5ad(a.h5ad, backed="r")
    n_cells = adata.n_obs
    print(f"dataset:   {n_cells:,} cells x {adata.n_vars:,} genes", flush=True)

    if a.use_raw:
        if adata.raw is None:
            sys.exit("--use_raw was requested but this h5ad has no .raw; "
                     "pass --no_use_raw only if .X already holds raw counts")
        X = adata.raw.X
        # `.raw.var_names` is not required to carry gene identifiers, and in the
        # scPlantDB build it does not: it holds the positional strings
        # '0','1','2',... while the identifiers live on `adata.var`. Reading
        # names off `.raw` there matches nothing, and -- worse -- a partial
        # match would attach one gene's threshold to another without any error.
        # Fall back to `adata.var_names` only when the two axes are the same
        # length, which is the one case where the positional correspondence is
        # well defined, and refuse otherwise.
        raw_names = np.asarray(adata.raw.var_names, dtype=str)
        looks_positional = not any(c.isalpha() for c in "".join(raw_names[:20]))
        if looks_positional:
            if adata.raw.n_vars != adata.n_vars:
                sys.exit(
                    f".raw carries positional names ({raw_names[:3].tolist()}) "
                    f"and its gene axis ({adata.raw.n_vars}) differs from "
                    f"adata.var ({adata.n_vars}), so the identifiers cannot be "
                    f"recovered safely. Re-export the h5ad with gene ids on "
                    f".raw.var, or pass --no_use_raw if .X holds raw counts.")
            gene_ids = np.asarray(adata.var_names, dtype=str)
            print("  .raw.var has positional names; taking gene ids from "
                  "adata.var (axes are the same length)", flush=True)
        else:
            gene_ids = raw_names
    else:
        X = adata.X
        gene_ids = np.asarray(adata.var_names, dtype=str)

    # ---- align the reference onto THIS dataset's gene axis, by identifier ----
    # ref_pos is every position whose gene the reference KNOWS, thresholded or
    # not; it defines the denominator.  keep_pos below is the narrower set that
    # actually carries thresholds and can emit a token.  The two differ by the
    # genes the reference saw but could not threshold (detected in < 50
    # pseudo-bulks), which belong in the library size and cannot be tokens.
    keep_pos, q_lo, q_hi, kept_ids = [], [], [], []
    for pos, g in enumerate(gene_ids):
        hit = ref.get(g)
        if hit is not None:
            keep_pos.append(pos)
            q_lo.append(hit[0])
            q_hi.append(hit[1])
            kept_ids.append(g)
    if not keep_pos:
        sys.exit("no gene identifier in this dataset matches the reference; "
                 "check that both use the same nomenclature (e.g. AT1G01010)")
    keep_pos = np.asarray(keep_pos, dtype=np.int64)
    ref_pos = np.asarray([pos for pos, g in enumerate(gene_ids) if g in universe],
                         dtype=np.int64)
    if ref_pos.size == 0:
        sys.exit("no gene here appears in the reference table at all")
    on_ref = off_ref = 0.0
    q_lo = np.asarray(q_lo, dtype=np.float64)
    q_hi = np.asarray(q_hi, dtype=np.float64)
    up_tok = np.array([f"{g}_UP" for g in kept_ids], dtype=object)
    dn_tok = np.array([f"{g}_DOWN" for g in kept_ids], dtype=object)

    cov = {
        "pooling": ("none, one sentence per sample" if a.k == 1
                    else f"k = {a.k}, {a.n_pools_div}-fold"),
        "down_rule": a.down_rule,
        "reference_genes": len(ref),
        "dataset_genes": int(len(gene_ids)),
        "matched_genes": int(len(keep_pos)),
        "dataset_genes_without_threshold": int(len(gene_ids) - len(keep_pos)),
        "reference_genes_absent_here": int(len(ref) - len(keep_pos)),
        "universe_genes": len(universe),
        "universe_genes_here": int(ref_pos.size),
        "genes_here_outside_reference": int(len(gene_ids) - ref_pos.size),
    }
    print(f"  matched {cov['matched_genes']:,} genes; "
          f"{cov['dataset_genes_without_threshold']:,} genes here have no "
          f"threshold (they emit nothing); "
          f"{cov['reference_genes_absent_here']:,} reference genes are absent "
          f"here", flush=True)
    if cov["matched_genes"] < 0.5 * len(ref):
        print("  WARNING: under half the reference matched. If identifiers "
              "look right, this dataset may simply be a narrow panel.",
              flush=True)

    vocab = None
    if a.vocab:
        vocab = {l.strip() for l in open(a.vocab) if l.strip()}
        print(f"vocabulary: {len(vocab):,} entries", flush=True)

    # ---- grouping: pool only within a label (and study, when given) ---------
    groups = defaultdict(list)
    if a.metadata:
        meta = pd.read_csv(a.metadata)
        for col in (a.cell_col, a.label_col):
            if col not in meta.columns:
                sys.exit(f"--metadata has no column {col!r}; "
                         f"it has {list(meta.columns)[:8]}")
        lab = dict(zip(meta[a.cell_col].astype(str),
                       meta[a.label_col].astype(str)))
        study = ({} if not a.study_col
                 else dict(zip(meta[a.cell_col].astype(str),
                               meta[a.study_col].astype(str))))
        obs_names = np.asarray(adata.obs_names, dtype=str)
        unlabelled = 0
        for i, bc in enumerate(obs_names):
            L = lab.get(bc)
            if L is None:
                unlabelled += 1
                continue
            groups[(L, study.get(bc, "NA"))].append(i)
        if unlabelled:
            print(f"  {unlabelled:,} cells carry no label and are skipped",
                  flush=True)
        if not groups:
            sys.exit("no cell matched the metadata; check --cell_col against "
                     "the h5ad obs_names")
    else:
        groups[("ALL", "NA")] = list(range(n_cells))
    print(f"  {len(groups)} (label, study) group(s)", flush=True)

    # ---- emit --------------------------------------------------------------
    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    f_sent = open(f"{out}_sentences.txt", "w")
    f_ids = open(f"{out}_cell_ids.txt", "w")
    f_meta = open(f"{out}_metadata.csv", "w", newline="")
    w_meta = csv.writer(f_meta)
    w_meta.writerow(["cell_id", "tissue"])

    rng = np.random.default_rng(a.seed)
    rng_shuf = np.random.default_rng(a.seed + 1)
    n_written = n_skipped = n_trunc = 0
    tok_total = tok_oov = 0
    up_tot = dn_tot = 0

    def safe(s):
        return "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(s))

    for (label, sf), idx in sorted(groups.items()):
        idx_arr = np.asarray(idx, dtype=np.int64)
        g = len(idx_arr)
        k_eff = min(a.k, g)
        n_pools = max(1, g // a.n_pools_div) if a.k > 1 else g
        for pool_i in range(n_pools):
            if a.k > 1:
                sampled = rng.choice(idx_arr, size=k_eff, replace=False)
            else:
                sampled = idx_arr[pool_i:pool_i + 1]
            # Row order matters for backed h5ad slicing: it must be increasing.
            rows = np.sort(sampled)
            pool_sum = np.asarray(X[rows].sum(axis=0)).ravel()
            # The library size is summed over the REFERENCE gene universe, not
            # over whatever this dataset happens to carry.  The corpus the
            # thresholds were estimated on summed over its own 53,678-gene axis,
            # which IS the reference universe, so for that corpus the two are
            # identical and the round-trip is unaffected.  They diverge for a
            # user who carries genes the reference never saw: counting those in
            # the denominator would shrink every normalised value relative to
            # the scale the thresholds live on, and the thresholds would then
            # mean something slightly different for that user than for us.
            # Genes the reference has and the user lacks contribute zero either
            # way.  `--lib_over_dataset_axis` restores the older behaviour.
            lib = pool_sum.sum() if a.lib_over_dataset_axis else pool_sum[ref_pos].sum()
            off_ref += float(pool_sum.sum() - pool_sum[ref_pos].sum())
            on_ref += float(pool_sum[ref_pos].sum())
            if lib <= 0:
                n_skipped += 1
                continue
            norm = np.log1p(pool_sum / lib * a.target_sum)

            v = norm[keep_pos]
            expressed = v > 0
            up_m = expressed & (v > q_hi)
            # 'nonzero' is the rule the shipped table was built under: a feature
            # at zero was excluded from its percentile and so cannot be DOWN.
            dn_m = (expressed & (v < q_lo)) if a.down_rule == "nonzero" \
                else (v < q_lo)
            toks = np.concatenate([up_tok[up_m], dn_tok[dn_m]])
            if len(toks) < a.min_tokens:
                n_skipped += 1
                continue
            up_tot += int(up_m.sum())
            dn_tot += int(dn_m.sum())

            rng_shuf.shuffle(toks)
            if len(toks) > a.max_tokens:
                toks = toks[:a.max_tokens]
                n_trunc += 1

            if vocab is not None:
                tok_total += len(toks)
                tok_oov += sum(1 for t in toks if t not in vocab)

            pb_id = f"PB_{safe(label)}_{safe(sf)}_{pool_i:06d}"
            f_sent.write(" ".join(toks) + "\n")
            f_ids.write(pb_id + "\n")
            w_meta.writerow([pb_id, label])
            n_written += 1

    for f in (f_sent, f_ids, f_meta):
        f.close()

    cov.update({
        # how much of this dataset's signal sits on genes the reference knows.
        # Well below 1 means the denominator, and therefore every threshold
        # comparison, is being set by material the reference never saw.
        "count_fraction_on_reference_genes":
            (on_ref / (on_ref + off_ref)) if (on_ref + off_ref) else None,
        "pseudobulks_written": n_written,
        "pseudobulks_skipped": n_skipped,
        "pseudobulks_truncated": n_trunc,
        "up_tokens": up_tot,
        "down_tokens": dn_tot,
        "up_down_ratio": (up_tot / dn_tot) if dn_tot else None,
        "oov_token_instances": tok_oov if vocab is not None else None,
        "oov_token_fraction": (tok_oov / tok_total) if (vocab and tok_total)
                              else None,
    })
    import json
    with open(f"{out}_coverage.json", "w") as fh:
        json.dump(cov, fh, indent=2)

    print(f"\nwrote {n_written:,} pseudo-bulks "
          f"({n_skipped:,} skipped under --min_tokens, {n_trunc:,} truncated)",
          flush=True)
    if dn_tot:
        print(f"  UP:DOWN = {up_tot/dn_tot:.3f}   "
              f"({up_tot:,} UP, {dn_tot:,} DOWN)", flush=True)
    if vocab is not None and tok_total:
        print(f"  outside the model vocabulary: {tok_oov:,} of {tok_total:,} "
              f"token instances ({tok_oov/tok_total*100:.3f}%) -> [UNK]",
              flush=True)
    print(f"  coverage written to {out}_coverage.json", flush=True)


if __name__ == "__main__":
    main()
