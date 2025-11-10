# 🧠 MyAgent — Building and Merging LoRA Models from Scratch

**Author:** [Clark Kitchen](https://www.linkedin.com/in/clarkkitchen)  
**Repository:** https://github.com/goldbar123467/RAG-LLAMA-3.2-Model-/
**Hugging Face Model:** [mistral-7b-lora-merged](https://huggingface.co/clarkkitchen22/mistral-7b-lora-merged)

---

## 🚀 Overview

**MyAgent** is a personal AI development project built entirely over a weekend — combining Python, Hugging Face tools, and local GPU workflows to **merge and deploy custom LoRA fine-tuned models**.  

The goal was to automate the process of merging lightweight adapters, testing inference locally, and publishing to the Hugging Face Hub. This project taught me how to move from idea → working AI toolchain — covering dataset management, model compatibility, and deployment automation.

---

## 🧩 Key Features

- 🔗 **LoRA Model Merging** — Combines multiple LoRA adapters into a single unified model.  
- ⚙️ **Adapter Offload Management** — Handles `meta` and offloaded layers seamlessly.  
- 🧰 **One-Click Upload** — Automated push to Hugging Face with `huggingface_hub`.  
- 🧮 **GPU-Optimized** — Built and tested locally on RTX hardware (PyTorch + CUDA).  
- 💾 **Safe Checkpointing** — Creates intermediate backups for rollback and testing.  
- 📊 **Readable, Modular Code** — Each component split into logical Python scripts.  

---

## 🧠 Technical Stack

| Component | Description |
|------------|--------------|
| **Language** | Python 3.12 |
| **Frameworks** | PyTorch, Transformers, PEFT |
| **Tools** | `huggingface_hub`, `accelerate`, `safetensors` |
| **Hardware** | NVIDIA RTX 2070 (CUDA 11.8) |
| **OS/Env** | Windows 10 + PowerShell (venv) |

---

## ⚡ Quick Start

Clone the repo and set up a virtual environment:
```bash
git clone https://github.com/clarkkitchen22/my_agent.git
cd my_agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
