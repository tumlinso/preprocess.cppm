# Numerical operations

The pinned comparison is Scanpy 1.11.5. Matching numerical behavior does not
require copying Scanpy's Python calling conventions. The public boundary uses
anndata arrays, explicit options, owned results, and ArrayView or ArrayEdit.
The source-version record is in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Data and ownership

Expression has shape observations by features. source(data, layer) acquires
X or the named layer for reading; edit_source acquires its mutable view. Typed
acquisition resolves dtype and layout before a numerical loop. Views retain
anndata storage lifetime according to the anndata contract.

Numerical overloads operate on matrix views directly. floating_copy explicitly
converts integer or floating input while preserving dense or sparse storage.
dense_copy explicitly materializes sparse or logical gather input. Transform
overloads accepting ArrayEdit mutate that buffer; their *_into counterparts
read a source and write a supplied destination. There is no implicit deep copy.

The implementation uses anndata's dense and sparse views directly, with local
pointers, dimensions and strides for each calculation. Preprocessing adds
operation-specific checks and calculations without another matrix descriptor
or lifetime model. Same-dtype dense copies delegate to anndata's `copy_into`.

Dense matrices support strides. Sparse inputs must be canonical CSR or CSC;
structural offsets and indices retain independent widths, and explicitly stored
zeros do not alter mathematical values. A gather without a direct addressable
view must be explicitly copied before processing. log1p preserves sparse
structure. Scaling without zero-centering preserves sparse structure; centered
sparse scaling requires an explicit dense destination because subtracting a
nonzero mean fills implicit zeros.

Integer matrices are valid for summaries and count-based operations. In-place
floating transforms accept FP32 or FP64 buffers. Convert integer input with
floating_copy, or use a caller-provided floating destination. Reductions
accumulate in FP64. The caller controls parallelism and synchronization.

## Quality control and filtering

qc reports cell and feature totals and detected counts; feature means and
dropout; optional log summaries; optional named feature-group totals and
fractions; and optional fractions of each observation total contained in its
top n features. Feature-group masks align with columns. Result arrays retain
observation or feature order. Totals, means, dropout and fractions use FP64;
detection counts use uint64. Set QcOptions::log1p false to omit log summaries.
Top-N fractions are opt-in; the default `percent_top` list is empty.

Filtering returns a boolean mask aligned to the selected axis. min_counts,
max_counts, min_detected, and max_detected may be combined; all supplied
bounds must pass. The result is a mask, not a hidden dataset mutation. Apply it
with anndata slicing, and call copy() when an independent owner is required.

feature_stats returns feature means and sample variances. An optional
observation mask selects rows for estimating those statistics.
An empty observation selection produces NaN statistics; one selected
observation has zero variance by the package's degenerate-input convention.

## Normalization and transformations

normalize_total divides each nonzero observation by its eligible total so it
reaches target_sum. By default, target_sum is the median positive eligible
observation total. An all-zero matrix remains unchanged; a zero-total row
remains zero. This gives the same default across dense, CSR, and CSC layouts,
including data with zero-total rows.
An all-zero input reports target 1 and zero factors.

Optionally, exclude_highly_expressed removes a feature from size-factor
calculation if it exceeds max_fraction of the original total in any
observation. The returned result contains the applied per-observation factors,
eligible-feature mask, and selected target.

The automatic target deliberately applies one rule across layouts. Scanpy 1.11.5
uses different all-zero handling paths for dense and sparse data; preprocess
uses the positive-total median in either layout. This difference is included in
independent qualification.

log1p computes the natural log(1+x) by default. An optional base changes the
logarithm base. annotate_log1p records the conventional AnnData uns metadata
explicitly when the corresponding layer has actually been transformed.

scale estimates per-feature means and sample standard deviations, optionally
from a row mask. By default it centers and scales to unit variance. Constant
features are retained and become zero when centered. Optional max_value clips
the scaled values. Set zero_center false to scale sparse values without
filling implicit zeros. apply_scale reuses an existing Scaling record on
another compatible matrix.
An observation mask restricts both parameter estimation and application to
selected rows; other rows remain unchanged.

## Highly variable features

highly_variable_genes supports HvgMethod::seurat and HvgMethod::cell_ranger.
Both methods use per-feature dispersion, bin genes by mean expression, and
normalize dispersion within each bin. Seurat expects logarithmized input and
converts back to the count scale using log_base when specified. Cell Ranger
computes dispersion directly on the supplied values without an inverse-log
transform, following Scanpy's implementation. n_bins defaults to 20; with n_top_genes, the
top-scoring selection is returned, otherwise mean and normalized-dispersion
cutoffs apply.

An optional batch-code array computes batch-aware selection summaries. The
result stays aligned to original feature order and includes the method, mean,
dispersion, normalized dispersion, selected mask, selected batch count, and
all-batches intersection. annotate_hvg attaches the supported results to var
and the method metadata to uns.

HVG statistics and annotations retain FP64 dispersion values. Scanpy's public
in-place path stores normalized dispersion in FP32, while its returned
non-in-place result retains the full precision values. The independent numeric
comparison uses that returned result; feature alignment and selected masks are
checked separately.

With top-gene selection in a single batch, every feature tied at the cutoff is
selected, so the final count may exceed n_top_genes. For batch-aware top-gene
selection, the global ranking orders by number of selected batches, then mean
normalized dispersion, and resolves remaining ties by stable original feature
order. ArrayView carries no feature labels. The explicit input order therefore
determines native tie results. Reference fixtures preserve feature ordering
when comparing masks.

Analytic and Scanpy comparison fixtures exercise tie handling, empty bins,
constant features, invalid input, and method options. See
[validation](validation.md).

## Dataset annotations

The annotate_* helpers explicitly move supported result arrays into the
corresponding obs, var, or uns slots. They validate axis lengths and duplicate
annotation names before mutation, and do not copy an entire AnnData object.
Low-level results carry no biological observation or feature ID map, so callers
must attach them to the same axis order used to compute them. A result can
instead be inspected, retained, or combined with caller-owned calculations
without modifying AnnData.
