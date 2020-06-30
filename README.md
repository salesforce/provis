# BERTology Meets Biology: Interpreting Attention in Protein Language Models

This repository is the official implementation of [BERTology Meets Biology: Interpreting Attention in Protein Language Models](https://arxiv.org/abs/2006.15222). 

## ProVis Attention Visualizer

![Image](images/vis3d_binding_sites.png?raw=true)  ![Image](images/vis3d_contact_map.png?raw=true)

### Installation
**General requirements**:
* Python >= 3.6
* PyTorch (See installation instructions [here](https://pytorch.org/).)

**Specific libraries:**
```
pip install biopython==1.77
pip install tape-proteins==0.4
```

**NGLViewer** (based on instructions found [here](https://github.com/arose/nglview#released-version])):

- Available on `conda-forge` channel

    ```bash
    conda install nglview -c conda-forge
    jupyter-nbextension enable nglview --py --sys-prefix
    
  # if you already installed nglview, you can `upgrade`
    conda upgrade nglview --force
    # might need: jupyter-nbextension enable nglview --py --sys-prefix
    ```

- Available on [PyPI](https://pypi.python.org/pypi/nglview/)

```bash
   pip install nglview
   jupyter-nbextension enable nglview --py --sys-prefix
```

To use in Jupyter Lab you need to install appropriate extension:

```bash
jupyter labextension install  nglview-js-widgets
```

### Execution

```
cd <project_root>/notebooks
jupyter notebook provis.ipynb
```

You may edit the notebook to choose other proteins, attention heads, etc. The visualization tool is based on the
excellent [nglview](https://github.com/arose/nglview) library.

---

## Experiments

### Installation

```setup
cd <project_root>
python setup.py develop
```


### Datasets

To download additional required datasets from [TAPE](https://github.com/songlab-cal/tape):

```setup
cd <project_root>/data
wget http://s3.amazonaws.com/proteindata/data_pytorch/secondary_structure.tar.gz
tar -xvf secondary_structure.tar.gz && rm secondary_structure.tar.gz
wget http://s3.amazonaws.com/proteindata/data_pytorch/proteinnet.tar.gz
tar -xvf proteinnet.tar.gz && rm proteinnet.tar.gz
```

### Attention Analysis

The following steps will recreate the reports currently found in <project_root>/reports/attention_analysis

Before performing steps, navigate to appropriate directory:
```
cd <project_root>/protein_attention/attention_analysis
```

#### Amino Acids

Run analysis (may wish to run in background):
```
sh scripts/compute_aa_features.sh
```
The above steps create a pickle extract file in <project_root>/data/cache

Run report from extract file:
```
python report_edge_features.py edge_features_aa
python report_aa_correlations.py edge_features_aa
```

#### Secondary Structure
Run analysis:
```
sh scripts/compute_sec_features.sh
```

Run reports:
```
python report_edge_features.py edge_features_sec
```
#### Contact Maps

Run analysis:
```
sh scripts/compute_contact_features.sh
```

Run report:
```
python report_edge_features.py edge_features_contact
```

#### Binding Sites
Run analysis:
```
sh scripts/compute_site_features.sh
```

Run report:
```
python report_edge_features.py edge_features_sites
```

#### Combined features
Create report of all features combined
```
python report_edge_features_combined.py edge_features_aa edge_features_sec edge_features_contact edge_features_sites
```

### Probing Analysis

The following steps will recreate the reports currently found in <project_root>/reports/probing


Navigate to directory:
```
cd <project_root>/protein_attention/probing
```

#### Training
Train diagnostic classifiers. Each script will write out an extract file with evaluation results. Note: each of these scripts may run for several hours.
```
sh scripts/probe_ss4_0_all
sh scripts/probe_ss4_1_all
sh scripts/probe_ss4_2_all
sh scripts/probe_sites.sh
sh scripts/probe_contacts.sh
```
#### Reports
```
python report.py
```

## License

This project is licensed under BSD3 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This project incorporates code from the following repo:
* https://github.com/songlab-cal/tape

## Citation

When referencing this repository, please cite [this paper](https://arxiv.org/abs/2006.15222).

```
@misc{vig2020bertology,
    title={BERTology Meets Biology: Interpreting Attention in Protein Language Models},
    author={Jesse Vig and Ali Madani and Lav R. Varshney and Caiming Xiong and Richard Socher and Nazneen Fatema Rajani},
    year={2020},
    eprint={2006.15222},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2006.15222}
}
```

