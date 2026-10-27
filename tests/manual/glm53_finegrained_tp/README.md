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
