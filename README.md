# Introduction

The preprint of the paper corresponding to this work can be found here: https://www.medrxiv.org/content/10.64898/2026.01.27.26344931v1

## abstract

Abstract
Digenic alterations can produce phenotypes such as synthetic lethality or digenic disease that are not observed upon individual gene perturbation, often by disrupting compensatory or redundant biological mechanisms. We hypothesized that gene pairs underlying such phenotypes share, when considered jointly, biological network properties analogous to those of essential genes or monogenic Mendelian disease genes. To test this hypothesis, we developed PAGAN, a graph representation learning framework that learns phenotypically relevant network signatures from single-gene labels in heterogeneous biological knowledge graphs and generalizes them to gene pairs without explicit pairwise supervision. PAGAN represents each gene pair as a new node embedded in the same network context as individual genes, enabling inference of pair-level properties from their combined neighborhood. Using multiplex knowledge-based networks in yeast and human, we show that PAGAN predicts synthetic lethal interactions and digenic disease gene pairs by training only on essential genes or monogenic disease genes, respectively. Across multiple evaluation settings, PAGAN achieves competitive performance relative to supervised state-of-the-art methods while avoiding reliance on currently limited and biased catalogs of known gene pairs. This framework provides a scalable strategy to explore combinatorial genetic architectures and prioritize candidate digenic interactions in functional genomics and rare disease diagnostics.

# Data download

Some files need to be downloaded from our google storage. Here are the links:

* https://storage.googleapis.com/pagan-cbl/human_knowledge_graph_20250620.tsv
This file should be put in the data/processed directory and be renamed human_knowledge_graph.tsv (or the name in the scripts should be updated)
* https://storage.googleapis.com/pagan-cbl/yeast_knowledge_graph_20240618.tsv
This file should be put in the data/processed directory and be renamed yeast_knowledge_graph.tsv (or the name in the scripts should be updated)
* https://storage.googleapis.com/pagan-cbl/yeast_pairs_20240617.tsv
This file should be put in the data/processed directory and be renamed yeast_pairs.tsv (or the name in the scripts should be updated)

* # Conda environment

The conda environment that was used during this work can be installed using the instructions in the conda_env.txt file
