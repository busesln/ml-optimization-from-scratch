# ML Optimization from Scratch

A Python project exploring gradient-based optimization on a Turkish question–answer relevance task. SGD, AdaGrad, RMSProp, and Adam are implemented with explicit tensor updates, using PyTorch autograd for gradients without `torch.optim`.

## Project Overview

- **Data:** Generate question–answer triplets through a local LLM, with separate training and test topics.
- **Embeddings:** Concatenate question and answer embeddings from `ytu-ce-cosmos/turkish-e5-large` into 2048-dimensional inputs, labelled `+1` or `-1`.
- **Models:** Train a single linear layer with `tanh`, or a two-layer MLP with ReLU and `tanh`, using mean squared error.
- **Experiments:** Configure GD, SGD, Adam, AdaGrad, and RMSProp runs across five random seeds and 100 epochs. GD uses the SGD update rule with a full batch.
- **Visualizations:** Plot training loss against epochs and estimated update counts, plus t-SNE projections of first-layer weight histories.

## Main Files

| File | Purpose |
| --- | --- |
| `ozel_optimizer.py` | Custom optimizer implementations |
| `modeller.py` | Single-layer and MLP models |
| `veri_uretici.py` | Synthetic dataset generation |
| `veri_vektorlestir.py` | Embedding generation and tensor datasets |
| `egitim.py`, `main_deney.py` | Training loop and experiment configuration |
| `analiz_ve_gorsellestirme.py` | Loss curves and weight trajectory plots |

## Setup and Usage

```bash
git clone https://github.com/busesln/ml-optimization-from-scratch.git
cd ml-optimization-from-scratch
python -m pip install torch sentence-transformers scikit-learn matplotlib seaborn numpy openai
```

Dataset generation requires LM Studio running at `http://localhost:1234/v1`, with the model identifier in `veri_uretici.py` matching the loaded model. The script is configured for `ytu-ce-cosmos/Turkish-Gemma-9b-T1`. Embedding generation requires downloading the E5 model.

After addressing the implementation notes below, run the pipeline in order:

```bash
python veri_uretici.py
python veri_vektorlestir.py
python main_deney.py
python analiz_ve_gorsellestirme.py
```

Choose the model when prompted. Experiments save `sonuclar1_Basit.pt` or `sonuclar1_MLP.pt`; set `DOSYA_ADI` in the plotting script accordingly. Figures are saved as PNG files.

## Implementation Notes

- `main_deney.py` currently passes `optimizer_adi="SGD"` for every run. Pass the computed `optimizer_adi` variable to compare the intended optimizers.
- The plotting script assumes 50 training samples, while preprocessing expects 100. Align dataset size and batch settings before interpreting update counts.
- Embedding generation currently repeats inside the record loop; move batch encoding after data collection. For CUDA training, move input and target tensors to the model's device.
