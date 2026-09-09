# Neural SDE Workflow

The detailed Neural SDE workflow is documented in the repository root README.

From the project root, use:

```bash
python experiments/neural_SDE/train_neural_SDE.py --trainer_cfg US_Stocks
python experiments/neural_SDE/generate_samples.py --trainer_cfg US_Stocks --mode mc --batch-size 10000
python experiments/neural_SDE/generate_samples.py --trainer_cfg US_Stocks --batch-size 1000
python experiments/neural_SDE/train_classification.py --trainer_cfg US_Stocks
```

See the top-level `README.md` for setup, generated artifacts, available CLI options, and the end-to-end workflow description.

