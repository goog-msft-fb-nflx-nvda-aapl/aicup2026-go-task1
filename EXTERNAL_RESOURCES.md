# External resources and AI assistance

## Accepted Task1 pretrained resource

- Source: KataGo, https://github.com/lightvector/KataGo
- Model: `b18c384nbt-humanv0.ckpt`; frozen checkpoint SHA256 `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`.
- Model release: https://github.com/lightvector/KataGo/releases/tag/v1.15.0
- Python source commit: `d91ea855110dae533f0aada947b2b7d78cc8a4e1`.
- Network notice: `licenses/KATAGO_NETWORK_LICENSE.txt`, source https://katagotraining.org/network_license/ . Source notice: `licenses/KATAGO_SOURCE_LICENSE.txt`. Preserve both notices when supplying the associated code/weights; third-party dependencies retain their own terms.
- Use: frozen policy-conditioned board features;432-dimensional actual-minus-expected residuals, learned identity MLP and LDA/Gaussian matching.
- Contribution evidence: the MLP.25 blend improved its then-fixed parent from.8630397 to.8666413. Replacing the older Gaussian by the expanded Gaussian on the later fixed LM4 fusion improved.8686865 to.8721950. These are specific comparisons, not an isolated causal attribution of the entire final score to HumanSL.

No extra raw Go dataset was introduced for the accepted policy experiments. The organizer's training/test files are not bundled. Pretrained-network internals are not represented as team-created weights.

## Software and methods

PyTorch supplies Transformer/MLP training and inference; NumPy/SciPy supply dense/sparse linear algebra; scikit-learn and joblib support legacy SVD/MLP feature processing and serialization; LightGBM supports historical gate/verifier components. The final new-pool path removes the last LambdaRank verifier. Exact installed versions are recorded in `evidence/final_candidates_environment.json`; source pins and notices remain authoritative. Transformer and ArcFace ideas are attributed to their papers in the report.

## Generative AI disclosure

Claude Code and OpenAI Codex assisted implementation, debugging, experiment execution, analysis and drafting. ChatGPT and Gemini were also used for research synthesis and experiment suggestions in the project workflow. TAN, CHUN-FEI directed the objective and experimental priorities and is responsible for reviewing the report and submission. Assistance is disclosed; no claim of unaided authorship is made. The team must verify the precise official contribution statement before final submission.

## Official disclosure guidance

The official FAQ requires a source URL, version/filename, licence and usage description for external resources, and recommends ablations of their contribution. The competition rules also require disclosure of generative-AI resources and their contribution.

Rules: https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688
FAQ: https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit
