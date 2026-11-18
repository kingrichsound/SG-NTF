# SG-NTF: Spectra-Guided Neural Tucker Factorization

[![Conference](https://img.shields.io/badge/ICCEA-2026-blue)](https://doi.org/10.1109/ICCEA69962.2026.11743580)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICCEA69962.2026.11743580-00629B)](https://doi.org/10.1109/ICCEA69962.2026.11743580)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

Official source code for the paper:

> **Spectra-Guided Neural Tucker Factorization**  
> Fusheng Wang and Yikai Hou  
> *2026 7th International Conference on Computer Engineering and Application (ICCEA)*, pp. 821–824, 2026.  
> Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11743580) | [DOI: 10.1109/ICCEA69962.2026.11743580](https://doi.org/10.1109/ICCEA69962.2026.11743580)

## Overview

**SG-NTF** is a neural tensor completion framework built upon low-rank Tucker decomposition, integrating:
- **Spectral / Fourier Feature Mapping** to alleviate spectral bias and capture multi-scale structural patterns across tensor modes;
- **Context-Aware Gating** to adaptively modulate latent mode interactions during neural Tucker reconstruction.

## Datasets & Availability

The benchmark tensors and train/validation/test splits evaluated in this work strictly follow the experimental protocol established in **Auto-Encoding Neural Tucker Factorization** ([Tang et al., *IEEE TKDE* 2025](https://doi.org/10.1109/TKDE.2025.3590198)).

> **Note:** As the preprocessed benchmark splits were curated by the authors of the upstream study and are not authorized for third-party redistribution, **this repository releases the model implementation only and does not host or distribute the dataset files** (please do not open issues or email this repository's maintainers requesting the raw/split data).  
> For dataset access or preprocessing inquiries, please refer to the original paper and contact the corresponding author(s) of **Tang et al. (*IEEE TKDE*, 2025)** directly.

## Citation

If you find this code useful in your research, please consider citing our paper (and the upstream benchmark paper for the experimental protocol):

```bibtex
@INPROCEEDINGS{wang2026sgntf,
  author    = {Wang, Fusheng and Hou, Yikai},
  booktitle = {2026 7th International Conference on Computer Engineering and Application (ICCEA)},
  title     = {Spectra-Guided Neural Tucker Factorization},
  year      = {2026},
  pages     = {821--824},
  doi       = {10.1109/ICCEA69962.2026.11743580}
}

@ARTICLE{tang2025autoencoding,
  author  = {Tang, Peng and Luo, Xin and Woodcock, Jim},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  title   = {Auto-Encoding Neural Tucker Factorization},
  year    = {2025},
  volume  = {37},
  number  = {10},
  pages   = {5795--5807},
  doi     = {10.1109/TKDE.2025.3590198}
}
```

## License

This project is released under the [MIT License](./LICENSE).
