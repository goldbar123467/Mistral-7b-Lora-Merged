# 🧠 LoRA Model Merging & Deployment System

**Author:** Clark Kitchen  
**LinkedIn:** [linkedin.com/in/clarkkitchen](https://www.linkedin.com/in/clarkkitchen)  
**Live Model:** [mistral-7b-lora-merged](https://huggingface.co/clarkkitchen22/mistral-7b-lora-merged) on HuggingFace

---

## What This Does

This project automates the process of **merging multiple LoRA fine-tuned models** into a single deployable model, then publishing it to HuggingFace Hub.

**In simple terms:** Instead of training a massive 7-billion parameter model from scratch (expensive, slow), LoRA lets you train tiny "adapter" layers (cheap, fast). This system takes multiple specialized adapters and combines them into one unified model.

**Why it matters:** You can fine-tune LLMs for specific tasks without needing a data center. Then merge the best versions together.

---

## Technical Overview

### The Problem
- Fine-tuning large language models (LLMs) is computationally expensive
- Training all 7B parameters requires significant GPU memory and time
- Organizations need domain-specific models but can't afford full retraining

### The Solution
**LoRA (Low-Rank Adaptation)** trains small adapter matrices instead of full weights:
- Reduces trainable parameters by ~90%
- Enables fine-tuning on consumer GPUs
- Multiple adapters can be merged post-training

**This system:**
1. Takes base model (Mistral-7B) + multiple LoRA adapters
2. Merges adapters into unified weight matrices
3. Handles GPU memory constraints and layer offloading
4. Publishes production-ready model to HuggingFace

---

## Architecture

```
Base Model (Mistral-7B)
    ↓
+ LoRA Adapter 1 (Task A)
+ LoRA Adapter 2 (Task B)  
+ LoRA Adapter 3 (Task C)
    ↓
Merge Algorithm
    ↓
Unified Model (All Tasks)
    ↓
HuggingFace Hub
```

### Key Components

**1. Adapter Loading (`load_adapters.py`)**
- Loads PEFT LoRA configurations
- Validates adapter compatibility with base model
- Handles `meta` device offloading for memory efficiency

**2. Merging Logic (`merge_loras.py`)**
- Implements weight combination strategies
- Options: averaging, weighted merge, SLERP interpolation
- Ensures numerical stability during merge

**3. GPU Optimization (`gpu_utils.py`)**
- Dynamic memory allocation
- Layer-wise processing for large models
- CUDA stream management for efficiency

**4. Deployment (`deploy.py`)**
- Creates model cards with metadata
- Uploads to HuggingFace Hub
- Generates usage examples and benchmarks

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Framework** | PyTorch 2.x |
| **Model Library** | Transformers (HuggingFace) |
| **Fine-tuning** | PEFT (Parameter-Efficient Fine-Tuning) |
| **Quantization** | bitsandbytes, safetensors |
| **Deployment** | huggingface_hub |
| **Hardware** | NVIDIA RTX (CUDA 11.8+) |
| **Language** | Python 3.10+ |

---

## Quick Start

### Prerequisites
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8 or higher
- Python 3.10+

### Installation

```bash
git clone https://github.com/goldbar123467/Mistral-7b-Lora-Merged.git
cd Mistral-7b-Lora-Merged
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Basic Usage

```python
from merge_loras import LoRAMerger
from transformers import AutoModelForCausalLM

# Initialize merger
merger = LoRAMerger(
    base_model="mistralai/Mistral-7B-v0.1",
    adapters=[
        "path/to/adapter1",
        "path/to/adapter2"
    ]
)

# Merge with weighted combination
merged_model = merger.merge(
    method="weighted",
    weights=[0.6, 0.4]
)

# Save merged model
merged_model.save_pretrained("./merged_output")

# Upload to HuggingFace
merger.push_to_hub(
    repo_id="username/model-name",
    token="your_hf_token"
)
```

---

## Merging Strategies

### 1. **Simple Average** (Default)
Averages adapter weights equally.
```python
merged_weight = (adapter1 + adapter2) / 2
```
**Use when:** Adapters have similar importance

### 2. **Weighted Merge**
Combines adapters with custom weights.
```python
merged_weight = w1 * adapter1 + w2 * adapter2
```
**Use when:** One adapter is more important

### 3. **SLERP (Spherical Linear Interpolation)**
Interpolates along the manifold of weight matrices.
```python
merged_weight = slerp(adapter1, adapter2, t=0.5)
```
**Use when:** Want smooth interpolation between adapters

---

## Performance Considerations

### Memory Management
- **Base Model:** ~14GB (FP16) or ~7GB (INT8)
- **Per Adapter:** ~100-500MB
- **Merging Peak:** ~20GB (loads all simultaneously)

**Optimization:** Use `device_map="auto"` for automatic layer offloading

### Speed Benchmarks
| Operation | Time (RTX 2070) |
|-----------|-----------------|
| Load base model | ~30 seconds |
| Load adapter | ~5 seconds |
| Merge 2 adapters | ~2 minutes |
| Save merged model | ~3 minutes |
| Upload to HF | ~5-10 minutes |

---

## Real-World Applications

### 1. **Domain-Specific LLMs**
Merge medical + legal adapters for healthcare compliance chatbot

### 2. **Multilingual Models**
Combine language-specific adapters into polyglot model

### 3. **Task Composition**
Merge summarization + translation adapters for cross-lingual summarization

### 4. **Continual Learning**
Add new capabilities without forgetting old ones

---

## Example: Mortgage Document Processing

**Scenario:** Build AI assistant for loan processing

**Approach:**
1. Fine-tune adapter on loan application documents
2. Fine-tune adapter on underwriting guidelines
3. Fine-tune adapter on fraud detection patterns
4. Merge all three for comprehensive mortgage AI

```python
merger = LoRAMerger(
    base_model="mistral-7b",
    adapters=[
        "./adapters/loan_docs",      # Weight: 0.4
        "./adapters/underwriting",   # Weight: 0.4
        "./adapters/fraud_detect"    # Weight: 0.2
    ]
)

mortgage_model = merger.merge(
    method="weighted",
    weights=[0.4, 0.4, 0.2]
)
```

**Result:** Single model that understands loan documents, applies underwriting rules, AND flags fraud.

---

## Technical Deep Dive

### How LoRA Works

Traditional fine-tuning updates all weights:
```
W_new = W_original + ΔW  (where ΔW is 7B parameters)
```

LoRA decomposes ΔW into low-rank matrices:
```
ΔW = A × B  (where A is 7B×r, B is r×hidden_dim, r << hidden_dim)
```

**Example:** 
- Full update: 7,000,000,000 parameters
- LoRA (r=8): ~50,000,000 parameters (99% reduction)

### Merging Algorithm

```python
def merge_lora_weights(base_model, adapters, weights):
    """
    Merge multiple LoRA adapters into base model
    
    For each layer:
        1. Extract base weight W
        2. For each adapter i:
            - Compute ΔW_i = A_i × B_i
        3. Combine: W_merged = W + Σ(weight_i × ΔW_i)
        4. Replace layer weight with W_merged
    """
    for layer_name in base_model.layers:
        base_weight = base_model.get_weight(layer_name)
        
        delta_total = 0
        for adapter, weight in zip(adapters, weights):
            A = adapter.get_matrix_A(layer_name)
            B = adapter.get_matrix_B(layer_name)
            delta = weight * (A @ B)  # Matrix multiplication
            delta_total += delta
        
        merged_weight = base_weight + delta_total
        base_model.set_weight(layer_name, merged_weight)
    
    return base_model
```

---

## Limitations & Future Work

### Current Limitations
- ⚠️ All adapters must use same base model
- ⚠️ Rank (r) must match across adapters
- ⚠️ High memory usage during merge (loading all adapters)
- ⚠️ No automatic weight optimization (manual tuning required)

### Planned Improvements
- [ ] Automatic weight search via grid search or Bayesian optimization
- [ ] Support for merging different-rank adapters
- [ ] Streaming merge for lower memory usage
- [ ] Integration with quantization (GGUF, GPTQ)
- [ ] Benchmarking suite for merge quality evaluation

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Use smaller batch size or enable CPU offloading
```python
merger = LoRAMerger(base_model, adapters, device_map="auto")
```

### "Adapter rank mismatch"
**Solution:** Ensure all adapters trained with same `r` parameter
```python
# Check adapter config
adapter.config.r  # Should match across all adapters
```

### "Model quality degraded after merge"
**Solution:** Try different merge weights or strategies
```python
# Experiment with weights
merger.merge(method="weighted", weights=[0.7, 0.3])  # Favor first adapter
```

---

## Contributing

Contributions welcome! Areas of interest:
- New merging algorithms (TIES, DARE, etc.)
- Memory optimization techniques
- Automated weight tuning
- Additional model architectures (LLaMA, Gemma, etc.)

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details

---

## Citation

If you use this code in your research or project, please cite:

```bibtex
@software{kitchen2024lora_merge,
  author = {Kitchen, Clark},
  title = {LoRA Model Merging and Deployment System},
  year = {2024},
  url = {https://github.com/goldbar123467/Mistral-7b-Lora-Merged}
}
```

---

## Acknowledgments

- HuggingFace team for Transformers and PEFT libraries
- Mistral AI for the base model
- LoRA paper: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

---

## Contact

**Clark Kitchen**  
Email: [your email]  
LinkedIn: [linkedin.com/in/clarkkitchen](https://www.linkedin.com/in/clarkkitchen)  
HuggingFace: [@clarkkitchen22](https://huggingface.co/clarkkitchen22)

---

**Built with 🧠 by someone who learned Python while building AI systems.**
