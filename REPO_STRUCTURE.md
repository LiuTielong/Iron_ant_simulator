# Repository Structure

The project is organised around three major components—ANT training, hardware simulation, and OliVe quantisation—along with supporting assets. The layout below lists the most relevant sub-directories and their purpose.

```
ANT-Quantization/
├── ant_quantization/          # ANT PyTorch implementation and experiment scripts
│   ├── antquant/              # Quantisation layers, utilities, and attention modules
│   ├── BERT/                  # NLP fine-tuning pipelines and GLUE tooling
│   ├── ImageNet/              # CV training/eval entrypoints and shell scripts
│   ├── quant/                 # CUDA extension for ANT quantisation kernels
│   └── result/                # Spreadsheet templates and recorded metrics
│
├── ant_simulator/             # Hardware performance & energy evaluation stack
│   ├── bitfusion/             # BitFusion-based accelerator models, benchmarks, and sweep tools
│   ├── dnnweaver2/            # Graph construction, tensor ops, and compiler utilities
│   ├── docs/                  # Simulator documentation and evaluation figures
│   └── results/               # CSV outputs produced by hardware experiments
│
├── olive_quantization/        # OliVe quantisation framework for LLMs
│   ├── antquant/              # OliVe-specific quantisation modules/utilities
│   ├── bert/                  # Scripts for GLUE/BART workloads
│   ├── llm/                   # GPT/BLOOM/OPT evaluation drivers and helpers
│   ├── quant/                 # CUDA kernels shared by OliVe experiments
│   └── figures/               # Paper figures and illustrative assets
│
├── results/                   # Top-level result spreadsheets aggregated across projects
├── README.md                  # Project overview, publications, and quick-start info
└── readme.ipynb               # Notebook summary of repository organisations
```

> Tip: each sub-project ships its own README with environment setup, dataset preparation, and reproduction scripts—start from those documents when working inside a specific module.

