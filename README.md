# M1 Memory Research

Author: **Megern Qaisse** ([GitHub](https://github.com/megern), [LinkedIn](https://www.linkedin.com/in/megernqaisse/)).

Two local studies on an Apple M1 with 16 GiB unified memory: a real 70B-checkpoint inference feasibility test, and small bilingual LoRA specialists with a retained pilot and a separate balanced confirmatory protocol.

The checksum-verified 70B checkpoint completed two local smoke executions. [The corrected report](results/llama70b-smoke-v2/report.json) records 164.71 seconds elapsed, first output at 131.33 seconds and 2.95 GiB sampled peak process RSS. Raw output is `Paris [end of text]`; the runtime adds its termination marker, so the original exact-whole-stdout check remains false. The first run and its duplicate-BOS warning are retained separately. A completed short smoke prompt is narrower than useful conversational speed or strong model quality. The manuscripts are editable drafts, not peer-reviewed publications.

## Completed local adaptation results

All three training seeds completed for both tasks. The same frozen 96-case test is used for each task's base and adapters. Classification permits only removal of an entire outer Markdown JSON fence; strict scores are separately retained in the raw reports.

| Synthetic task | Base accuracy | Adapter seeds 17 / 23 / 41 | Mean adapter macro-F1 |
|---|---:|---:|---:|
| Authentication rules | 33.33% | 72.92% / 72.92% / 83.33% | 0.761 |
| Workflow rules | 55.21% | 100% / 100% / 100% | 1.000 |

All adapters produced valid strict JSON for all cases; the base wrapped JSON in Markdown. Training took 180–221 seconds per adapter; peak MLX allocation was 561–609 MiB, which is not total process RAM. Each adapter weight file is 2,628,325 bytes and still requires the shared base. Seed 17 is the reference release, not the best score selected from test results.

See [the all-seed numerical summary](results/confirmatory-summary.json), [the figure](figures/specialist-accuracy.png), and [the exploratory error audit](results/exploratory-error-analysis.json). Authentication still fails on some threshold cases; its class-balanced generator has a language–label association. A language-only shortcut learned on training data scores 47/96, below all adapters but above the global majority reference. This audit was added after evaluation. Synthetic scores do not establish general Arabic or operational security competence.

The exported seed-17 adapters also completed a paired identity audit: anonymous account/source fields changed one authentication prediction (71/96 versus 70/96); an anonymous job changed no workflow prediction (96/96). These reuse the same semantic test cases and are exploratory robustness checks. The handcrafted `examples/auth.json` demo expects `high` by the stated rule but the reference 0.6B adapter actually returns `low`; this failure is retained. Both exported adapters load and run; loadability does not guarantee a correct classification.

A [later exploratory size-comparison protocol](data/size-comparison-protocol.json) uses a pinned Qwen3 1.7B 4-bit base for authentication, retaining all seeds and the same previously scored data. The base scored 39/96; adapter seeds 17/23/41 scored 73/96, 69/96 and 65/96. Their mean of 71.88% is below the 0.6B mean of 76.39%, despite greater time and memory allocation. See [all retained results](results/size-comparison-summary.json).

## 70B inference

The selected checkpoint is Llama 3.3 70B Instruct IQ2_XXS, pinned in [the manifest](models/llama70b.json). Its 19,097,390,048 bytes exceed 16 GiB of physical memory. The experiment uses existing llama.cpp CPU mmap behavior and OS paging. This first version has no custom base-layer streaming algorithm or algorithmic novelty claim.

The runtime is the official `llama-b11457-bin-macos-arm64.tar.gz` archive with SHA256 `e234070cbde0c8b0d30f79fa08ff8246abe22c72acb752619349b8d9c46f7839`. Download it from [the pinned official release](https://github.com/ggml-org/llama.cpp/releases/tag/b11457), verify that digest, and extract locally. Dependency installation and model-file retrieval require network access; every training, forward pass and evaluation in this project uses local files and local compute.

```sh
python3 -m pip install -r requirements.txt
python3 tools/install_runtime.py --output runtime
python3 tools/download_model.py --manifest models/llama70b.json --output /absolute/local/models/llama70b.gguf --workers 4
python3 tools/run_large_model.py --runtime runtime/llama-b11457/llama-completion --model /absolute/local/models/llama70b.gguf --manifest models/llama70b.json --output results/my-70b-run --timeout 900 --tokens 8
```

Use fresh output directories. The executor verifies weights before starting, disables remote inference and GPU offloading, limits context to 256, and records raw output, library timings and process-tree memory. It stops its own process group on excessive memory/swap growth or timeout. See the [readable inference manuscript](papers/inference-study.md) and [editable LaTeX source](papers/inference-study.tex) for guards, analytical bounds and limits. CPU mmap permits a virtual mapping larger than RAM; it does not imply that all weights are resident or that generation is fast.

The [memory figure](figures/inference-memory.png) uses the corrected run. The pinned GGUF [header inspection](results/llama70b-inspection.json) records 80 blocks and approximately 70.55 billion stored tensor elements. Process RSS is not model file size or total system RAM; disk-backed pages are continually reclaimed. One decoding run does not establish sustained speed. Existing system swap was present; no positive growth was observed during these measured runs.

`results/runtime-control` is an executor check using a tiny stories model. Its generated text is neither a 70B result nor a quality benchmark. Model weights and runtime binaries are excluded from this source tree.

## Small specialists

The base is [Qwen3 0.6B 4-bit at the pinned revision](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/tree/73e3e38d981303bc594367cd910ea6eb48349da8). Acquire its config, tokenizer and safetensors files locally first. Training/evaluation tools verify the base config and weight hashes before attributing results to that checkpoint. No model identifiers trigger automatic downloads.

Two task families share this base:

- `auth`: classify explicit authentication-event counts and time windows.
- `workflow`: classify explicit GitHub workflow-policy properties.

The original pilot is preserved under `data/auth`, `data/workflow` and the corresponding result files. It exposed majority collapse in the authentication adapter and a JSON-format confound in the base. A separate [confirmatory protocol](data/confirmatory-v2/protocol.json) was frozen before training: balanced labels, 384 training/48 validation/96 test cases per task, three training seeds, 400 iterations and adapters in the last 16 layers. Account/job identities and instruction surfaces are split; authentication test time values include unseen boundary conditions. Underlying rules remain shared, so these are synthetic stated-rule tasks, not broad generalization tests.

```sh
python3 tools/download_base.py --output /absolute/local/qwen3-0.6b-4bit
python3 tools/train_specialist.py --model /absolute/local/qwen3-0.6b-4bit --data data/confirmatory-v2/auth --output artifacts/auth-seed17 --iters 400 --layers 16 --seed 17
python3 tools/evaluate_specialist.py --model /absolute/local/qwen3-0.6b-4bit --data data/confirmatory-v2/auth/test.jsonl --output results/auth-base-local.json
python3 tools/evaluate_specialist.py --model /absolute/local/qwen3-0.6b-4bit --data data/confirmatory-v2/auth/test.jsonl --adapter artifacts/auth-seed17 --output results/auth-adapter-local.json
python3 -m unittest discover -s tests -v
```

Repeat with `workflow` and seeds 23/41 for the frozen confirmatory series. Use fresh outputs. The final checkpoint is scored without selecting against test accuracy. Reports retain strict schema accuracy and classification after removing only a complete outer JSON fence; arbitrary prose extraction and label repair are excluded. Macro-F1 exposes majority prediction. See [the readable specialist manuscript](papers/specialist-study.md) and [editable LaTeX source](papers/specialist-study.tex).

To run the complete original series sequentially, use `python3 tools/run_experiments.py --model /absolute/local/qwen3-0.6b-4bit --series confirmatory --output runs/fresh-reproduction`. The runner verifies all frozen split hashes before training, keeps all three seeds and writes fresh reports without overwriting the recorded study. It never downloads or executes remotely. The later comparison uses `--series size1p7` with the verified 1.7B base, obtained explicitly through `download_base.py --manifest models/qwen3-1.7b.json --output /absolute/local/qwen3-1.7b-4bit`.

Regenerate data into a fresh location with `python3 tools/make_confirmatory_data.py --output /absolute/local/fresh-data`; compare split hashes with the recorded protocol. The generator refuses to overwrite the original freeze. After every evaluation completes, run `tools/summarize_study.py --output results/fresh-summary.json` and `tools/export_adapter.py --task auth --output /absolute/local/fresh-adapter`. Export checks evaluated weight and test hashes and removes machine-specific paths from the adapter config. Repeat for `workflow`.

For exportable plots, install `requirements-figures.txt` and run `tools/plot_results.py --summary results/confirmatory-summary.json --output figures`. The saved figure displays all seeds, rather than selecting one favorable trial.

For a single local prediction, run `python3 tools/predict.py --model /absolute/local/qwen3-0.6b-4bit --adapter /absolute/local/exported-adapter --task workflow --record examples/workflow.json --language ar`. Replace the task and example for authentication. The command validates required evidence fields, uses the recorded prompt format and returns the raw output alongside its strict label. New inputs remain experimental; this interface does not guarantee correct decisions.

These are small adapters over one foundation model, not independently pretrained models. A 4 GiB MLX allocation guard is not a hard total-RAM cap. Traces, generated labels, runtime versions, data hashes and limitations accompany the study. Rule-based tools can solve these explicit rules directly; a good synthetic score does not establish production SOC, Arabic language, or security capability.

## العربية

[دليل التشغيل المحلي بالعربية](OPERATING_GUIDE.md) يشرح تنزيل الأوزان، إعادة التدريب، تجربة المحوّلات، وحدود تشغيل 70B.


ندرس إمكانية تشغيل نموذج 70B حقيقي عبر القرص على ماك M1 بذاكرة 16 GiB، ونقيس الاستجابة والذاكرة بدل الاكتفاء بتحميل الملف. وبشكل منفصل ندرب محوّلات صغيرة لمهمتين محددتين، ونقارنها بالنموذج الأصلي على حالات محجوزة بالعربية والإنجليزية. نحتفظ بنتائج التجربة الأولية والنتائج السلبية، ونفصل صحة التصنيف عن صحة التنسيق. التدريب والتقييم محليان بالكامل؛ تنزيل الملفات العامة هو خطوة الشبكة اللازمة فقط.

## Materials and licenses

Own tools and synthetic data: MIT. Base Qwen weights: Apache-2.0. Llama weights: Llama 3.3 license, not relicensed by this project's source license. Neither base is redistributed here. Paper drafts are author-review materials; no journal, institution, review status or novelty is implied.

Primary research: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [LLM in a flash](https://arxiv.org/abs/2312.11514), [FlexGen](https://arxiv.org/abs/2303.06865).

## Exploratory policy projection

The new `normalized-auth` prototype moves count/time comparisons into explicit Python code and trains the 0.6B adapter on Boolean policy facts. Data is balanced within each language. It has only eight possible states, and a direct rule baseline can solve the whole task. Sampling and wording also differ from the raw-numeric series, so any gain cannot be attributed to one change or to LLM arithmetic. Seeds 17/23/41 scored 96/96, 96/96 and 64/96: mean 88.89%, macro-F1 mean 0.852. The third seed labels every medium case high. The base scored 47/96 after outer-fence removal; direct rule code scored 96/96. All adapters produced strict JSON. Each trained in 202–203 seconds. See [the all-seed summary](results/normalized-summary.json) and [figure](figures/normalized-accuracy.png). Reproduce with `run_experiments.py --series normalized`; single predictions require `predict.py --normalize --task auth`.

## Public reference adapters

Seed 17 was specified as the reference, without best-test-score selection. Each release contains only the evaluated adapter, a portable configuration, hashes, license and model card. Shared base weights are acquired separately.

- [Authentication: raw evidence](https://huggingface.co/megern/m1-auth-rule-adapter)
- [Workflow policy](https://huggingface.co/megern/m1-workflow-rule-adapter)
- [Authentication: exploratory 1.7B](https://huggingface.co/megern/m1-auth-1p7-adapter)
- [Authentication: Python feature projection](https://huggingface.co/megern/m1-auth-normalized-adapter)
