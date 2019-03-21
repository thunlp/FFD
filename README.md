# Fact Discovery from Knowledge Base via Facet Decomposition
This repo contains the source code and dataset for the following paper:
Fact Discovery from Knowledge Base via Facet Decomposition. Zihao Fu, Yankai Lin, Zhiyuan Liu and Wai Lam. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019).

## How to use our code for FFD

### Prerequisite
- g++ 7.3.0
- Python 2.7.16

All the codes are tested under Ubuntu 18.04.1 LTS.

### Dataset
We made a new dataset based on FB15k, it was already in `ANALOGY/FB15k`. The dataset structure is as follows:

    FB15k
    ├── p0.5-entities.txt
    ├── p0.5-relations.txt 
    ├── p0.5-test.txt
    ├── p0.5-train.txt
    ├── p0.5-valid.txt
    ├── entity2id.txt
    └── relation2id.txt

in which, `p0.5-train.txt`, `p0.5-test.txt`, `p0.5-valid.txt` are the tain, test and valid set respectively. Each line contains the head, relation and tail of a fact. `p0.5-entities.txt` and `p0.5-relations.txt` contains the name of all entities and relations. `entity2id.txt` and `relation2id.txt` contain ids for entities and relations.



### Usage
1. Clone
```
git clone https://github.com/fuzihaofzh/FFD.git
```

2. Compile & Install
```
cd ANALOGY
make
cd ..
pip install -r requirements.txt
```

3. Train Entity-relation Facet Component & Tail Inference Facet Component
```
python facts_discovery.py run --inputTag p0.5 --cudaId 0 --step trainCorNet
ANALOGY/main -algorithm Analogy -model_path output/Analogy_FB15k_p0.5.model -dataset ANALOGY/FB15k/p0.5 -num_thread 8
```

4. Train FFD and predict
```
python facts_discovery.py run --inputTag p0.5 --cudaId 0 --step feedback
```

### Cite

    @inproceedings{fu2019fact,
      title={Fact Discovery from Knowledge Base via Facet Decomposition},
      author={Fu, Zihao and Lin, Yankai and Liu, Zhiyuan and Lam, Wai},
      booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
      volume={1},
      year={2019}
    }





