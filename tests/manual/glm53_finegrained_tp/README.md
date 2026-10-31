# GLM-5.3-Flash fine-grained TP correctness patch

This temporary validation patch records the exact behavior used by the
GLM-5.3-Flash A3 W8A8 experiments.

- `vllm_ascend/utils.py` keeps target OProj/MLP fine-grained TP inside normal
  DP coordination so every rank uses matching token rows, graph selection and
  padding. Draft/MTP coordination policy is unchanged.
- `vllm_ascend/worker/v2/model_runner.py` makes an idle target rank issue the
  Embedding collective at the same input-preparation point as active ranks.
- `main_only_scope.py` is a test-only adapter. It disables all fine-grained TP
  knobs while constructing and executing MTP. For Embedding and LM Head it
  prevents target/draft module aliasing, reloads the original TP2 checkpoint
  slice, and verifies the loaded bytes before a request is run.

The adapter is not installed by the package and must be imported and installed
by the validation launcher before either GLM5Next MTP module is imported.

## Target prefill OProj and dense MLP

`p_prefill_uniform_1031.py` records the additional adapter used by the
16-die DP16/TP1 prefill experiments. Install it in each worker after the
normal vLLM 0.31 compatibility dispatch patch and before model execution.
Use `FG_ROLE=p`, `P_FORCE_FG_ROWS=1`, and `P_ZERO_FG_PADDING=1`.
The existing target-only MTP scope must also be installed. Diagnostic flags
`P_FINITE_PROBE`, `P_REFERENCE_PROBE`, and `P_UNIFORM_TARGET` are disabled
for performance and accuracy measurements.

The DP descriptor already agreed on the maximum token rows, but the MLA
backend passed only its local real rows to OProj. Consequently the
AllToAll and ReduceScatter payloads had inconsistent row counts. The
adapter pads at the projection entry, zeros dummy rows, and trims the
returned tensor to its original local rows. Dense MLP keeps this original
row count across its column/row projections and trims after the row projection.

The execute guard excludes graph capture and warmup. This prevents a stale
16K prefill descriptor from padding a 48-token graph warmup. Draft execution
retains its original fine-grained configuration, so the adapter is inactive
inside the existing MTP capture/propose calls.

The adapter uses process-local state: one worker runs one synchronous target
forward at a time, with draft calls nested inside it. It adds no collective.
It wraps the inherited dispatch because this version has no hook to replace
the eager batch descriptor after DP agreement. A permanent implementation
should carry row counts through the runner/forward context and pad inside
the custom projection operators, replacing this validation adapter.

Hardware regression coverage includes four idle-peer requests, 16 uneven
prefill requests, 4K/1 at eight requests per rank and 128K/1 at two requests
per rank. Both projection adapters complete the uneven functional requests;
full GPQA and matched-budget performance results are recorded in the wiki.
`p_projection_reference_1031.py` optionally compares three OProj layers
against the original full checkpoint; do not retain its extra weights in
performance runs.
