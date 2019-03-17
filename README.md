# FDD
This repo contains codes accompanying paper "Fact Discovery from Knowledge Base via Facet Decomposition" (NAACL-HLT 2019).

# Quick Start
1. Clone
```
git clone https://github.com/fuzihaofzh/FFD.git
```

2. Compile Analogy
```
cd ANALOGY
make
cd ..
```

3. Train Entity-relation Facet Component & Tail Inference Facet Component
```
python facts_discovery.py run --inputTag p0.5 --cudaId 0 --step trainCorNet
ANALOGY/main -algorithm Analogy -model_path output/Analogy_FB15k_p0.5.model -dataset ANALOGY/FB15k/p0.5
```

4. Train FFD and predict
```
python facts_discovery.py run --inputTag p0.5 --cudaId 0 --step feedback
```

# Dataset
We made a new dataset based on FB15k, it was already in `ANALOGY/FB15k`.

# Cite
Please Cite: Fact Discovery from Knowledge Base via Facet Decomposition
Zihao Fu, Yankai Lin, Zhiyuan Liu and Wai Lam


