# 3DGS Dynamic Masking: Static Artifact Reduction

## Overview
This project demonstrates that integrating 2D semantic masks into 3D Gaussian Splatting (3DGS) training eliminates static background artifacts caused by dynamic robot motion. We achieve 107x static L1 error reduction through two strategies: loss masking and ray filtering.

## Key Results
| Method | Static L1 Error | Improvement |
|--------|-----------------|-------------|
| Baseline (Unmasked) | 0.1067 | - |
| Loss-Mask | 0.0000 | 107x |
| Ray-Filter | 0.0000 | 107x |

## Project Structure
3dgs-dynamic-masking/
├── data/generated/ - PyBullet synthetic dataset
├── output/ - Model checkpoints (baseline, loss_mask, ray_filter)
├── results/ - Final tables and figures
└── notebooks/ - Full pipeline (Colab)

## Quick Start
1. Environment: pip install -e gaussian-splatting
2. Dataset: Run Section 1 (PyBullet generation)
3. Train: Run Sections 2-4 (baseline, loss-mask, ray-filter)
4. Compare: Run Section 5 (quantitative analysis)

## Method Details

### Loss Masking
Modify training loss to supervise only static background:
L1_masked = mean(|render - GT| * (1 - mask))

### Ray Filtering
Skip rays projecting into dynamic regions during rasterization.

## Key Findings
- Both masking strategies eliminate static artifacts
- Loss masking simpler and preferred
- 16-Gaussian models, no densification, fair comparison
- Perfect static reconstruction (L1=0) achieved

## Limitations
- Synthetic PyBullet dataset
- Depends on high-quality semantic masks
- Fixed 16-Gaussian constraint

## Future Work
- Real robotic datasets
- Temporal consistency
- End-to-end mask learning
- Integration with 4D-GS

## References
Kerbl et al. (2023). 3D Gaussian Splatting for Real-time Radiance Fields
https://github.com/graphdeco-inria/gaussian-splatting

Project Status: Complete
