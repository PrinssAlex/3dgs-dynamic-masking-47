
# Increasing the Robustness of Gaussian Splatting by Masking Dynamic Objects

## Participants

*   Prince Ogbodum [474343]
*   Jacob Ahemen [476149]
*   Christopher Mandengs [486554]

## Description of the Conducted Research

This project investigates methods to improve the robustness of 3D Gaussian Splatting (3DGS) when dealing with dynamic scenes containing moving objects. Standard 3DGS struggles in such scenarios, often producing artifacts like ghosting and blurring. To address this, we explored two primary strategies for integrating 2D semantic masks into the 3DGS training pipeline to suppress reconstruction artifacts caused by dynamic objects:

1.  **Loss Masking**: This approach involves modifying the loss function to ignore the regions of the image that contain dynamic objects. The L1 loss is multiplied by a binary mask where static pixels have a value of 1 and dynamic pixels have a value of 0. This way, the model is not penalized for errors in the dynamic regions, effectively learning to ignore them.

2.  **Ray Filtering**: This is a more direct approach where the rendering process itself is altered. During the rasterization of the Gaussians, we filter out the rays that would project to pixels marked as dynamic in the 2D mask. This prevents the dynamic objects from ever being incorporated into the 3D scene representation.

The project's hypothesis is that both methods will significantly reduce artifacts and improve the photorealism of the reconstructed scenes compared to an unmasked baseline. We evaluated the effectiveness of these two strategies by comparing them against a baseline 3DGS model trained on the same dataset without any masking. The evaluation was performed quantitatively using metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS), and qualitatively by inspecting the rendered images for artifacts.

## Demonstration (Video)

A video demonstrating the results of our project can be found here: (https://github.com/jacob936/Improving-the-Robustness-of-Gaussian-splatting-by-masking-dynamic-Objects/blob/main/Image%20and%20Videos/VID-20251210-WA0017.mp4)

You can also find a GIF animation of the results below:

![GIF of results](https://github.com/PrinssAlex/3dgs-dynamic-masking-47/blob/main/Image%20and%20Videos/images.gif)
![GIF of results](https://github.com/PrinssAlex/3dgs-dynamic-masking-47/blob/main/Image%20and%20Videos/masks.gif)

## Installation and Deployment

This project was developed and executed on **Google Colab Pro** using a **T4 GPU**.

### Step-by-step instructions to set up the environment:

1.  **Clone the 3D Gaussian Splatting repository:**

    ```bash
    git clone https://github.com/graphdeco-inria/gaussian-splatting
    ```

2.  **Navigate to the cloned directory:**

    ```bash
    cd gaussian-splatting
    ```

3.  **Install the required Python dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python ultralytics matplotlib scikit-learn lpips
    pip install -e .
    ```

## Running and Usage

Here are the instructions to run the different parts of the project.

### How to start model training:

*   **Baseline Model (no masking):**

    ```bash
    python train.py -s /3dgs-dynamic-masking-47/data/generated -m /3dgs-dynamic-masking-47/output/baseline --iterations 5000
    ```

**Strategy 1: Loss Masking:**

    ```bash
    python train_loss_mask.py -s /3dgs-dynamic-masking-47/data/generated -m /3dgs-dynamic-masking-47/output/loss_mask --iterations 5000 --mask_dir c
    ```

**Strategy 2: Ray Filtering:**

    ```bash
    python train_ray_filter.py -s /3dgs-dynamic-masking-47/data/generated -m /3dgs-dynamic-masking-47/output/ray_filter --iterations 5000 --mask_dir /3dgs-dynamic-masking-47/data/generated/masks
    ```
    *Note: The training scripts `train_loss_mask.py` and `train_ray_filter.py` are modified versions of the original `train.py` script from the 3DGS repository.*

## THEORETICAL BACKGROUND:

## 3D Gaussian Splatting model

A 3D Gaussian $\mathcal{G}_i$ is defined by mean $\mu_i \in \mathbb{R}^3$, covariance $\Sigma_i \in \mathbb{R}^{3 \times 3}$, color $c_i \in \mathbb{R}^3$ and opacity $\alpha_i \in [0,1]$. For a camera with projection $\Pi$, the Gaussian is approximated in the image plane as a 2D Gaussian[^1][^2]

$$
\tilde{\mathcal{G}}_i(u) = \exp\!\left(-\tfrac{1}{2}(u - \tilde{\mu}_i)^\top \tilde{\Sigma}_i^{-1} (u - \tilde{\mu}_i)\right),
$$

where $\tilde{\mu}_i$ and $\tilde{\Sigma}_i$ are obtained by projecting $(\mu_i,\Sigma_i)$ through the camera and linearizing the projection.[^1]

Given all Gaussians intersecting a pixel $u$, sorted by depth along the ray, the pixel color is computed by front-to-back alpha compositing

$$
C(u) = \sum_{i} T_i(u)\,\alpha_i(u)\,c_i,
$$

with per-Gaussian contribution $\alpha_i(u) = 1 - \exp(-\tau\,\tilde{\mathcal{G}}_i(u))$ and transmittance

$$
T_i(u) = \prod_{j < i} \bigl(1 - \alpha_j(u)\bigr),
$$

where $\tau$ is a scaling factor controlling opacity.

## Photometric loss without masking

Given a ground-truth image $I_{\text{gt}}$ and rendered image $I_{\theta}$ (parameters $\theta$ are all Gaussian attributes), a standard photometric loss is

$$
\mathcal{L}_{\text{photo}} = \lambda_{1}\,\|I_{\theta} - I_{\text{gt}}\|_{1}
 \lambda_{\text{ssim}}\,\bigl(1 - \text{SSIM}(I_{\theta}, I_{\text{gt}})\bigr),
$$

where $\lambda_1$ and $\lambda_{\text{ssim}}$ weight the L1 and SSIM components respectively.[^3][^4]

## Loss masking with static/dynamic masks

Let M(u) \in \{0,1\} be a binary mask for pixel u, where M(u)=1 denotes static background and M(u)=0 denotes dynamic regions to be ignored. The masked photometric loss can be written as

$$
\mathcal{L}_{\text{mask}} =
\frac{\sum_{u} M(u)\,\bigl|I_{\theta}(u) - I_{\text{gt}}(u)\bigr|}
{\sum_{u} M(u) + \varepsilon},
$$

optionally combined with SSIM only over static pixels

$$
\mathcal{L}_{\text{total}} =
\lambda_{1}\,\mathcal{L}_{\text{mask}}
 \lambda_{\text{ssim}}\,\bigl(1 - \text{SSIM}(I_{\theta}\odot M, I_{\text{gt}}\odot M)\bigr),
$$

where $\odot$ is elementwise multiplication and $\varepsilon$ avoids division by zero.[^5][^6]

The dynamic pixels contribute zero to the loss, so the corresponding Gaussians receive no gradient from those regions.

## Ray filtering with dynamic masks

For ray filtering, define the set of image pixels whose masks are static:

$$
\Omega_{\text{static}} = \{u \mid M(u) = 1\}.
$$

During training, only rays corresponding to $\Omega_{\text{static}}$ are traced and used in the loss:

$$
\mathcal{L}_{\text{ray}} =
\frac{1}{|\Omega_{\text{static}}|}
\sum_{u \in \Omega_{\text{static}}}
\bigl|I_{\theta}(u) - I_{\text{gt}}(u)\bigr|.
$$

Equivalently, for each Gaussian $\mathcal{G}_i$, you can mask out its contribution for dynamic pixels by redefining the effective opacity as

$$
\alpha_i^{\text{eff}}(u) = M(u)\,\alpha_i(u),
$$

and using $\alpha_i^{\text{eff}}$ in the alpha compositing equations above.[^7][^8]

This formulation makes the connection between “skipping Gaussians on dynamic pixels” and the underlying alpha compositing explicit.

## Evaluation metrics

You can also explicitly write the metrics you already compute:

- PSNR between $I_{\theta}$ and $I_{\text{gt}}$:

$$
\text{PSNR} = 10 \log_{10} \left(
\frac{\text{MAX}^2}{\text{MSE}(I_{\theta}, I_{\text{gt}})}
\right),
$$

where MAX is the maximum possible pixel value and MSE is mean squared error.[^9]

- SSIM averaged over the image:

$$
\text{SSIM}(x, y) =
\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},
$$

computed over local windows and averaged.[^4]

- LPIPS as a learned perceptual distance in a deep feature space between $I_{\theta}$ and $I_{\text{gt}}$.[^5][^3]

The folder is structured as follows:
### Project Structure
 ```
data/
├──pybullet_dataset
├──raw_dataset

gaussian-splatting/
├──assets
├──.git
├──arguments
├──gaussian_renderer
├──IpipsPyTorch
├──utils
├──submodules
├──SIBR_viewers

3dgs-dynamic-masking/
├── data/generated/          # PyBullet dataset-Contains the synthetic dataset used for training and evaluation.
│   ├── images/             # 30 RGB frames
│   ├── masks/              # Binary masks
│   └── camera_poses.json   # Camera parameters
├── output/       # Training results- Contains the output of the training runs for the baseline, loss masking, and ray filtering models.
│   ├── baseline/           # Unmasked training- Output for the baseline model.
│   ├── loss_mask/          # Loss masking -Output for the loss masking model.
│   └── ray_filter/         # Ray filtering -Output for the ray filtering model
├── results/                 # Metrics & figures - Contains the final comparison results.
│   ├── metrics_comparison.csv - A CSV file with the PSNR, SSIM, and LPIPS scores for all three models.
│   ├── figure_psnr_comparison.png - Side-by-side comparison images of the rendered views.
│   └── figure_l1_error_comparison.png

├── notebooks/               # Colab notebook
├── images_and_videos
├── requirements.txt         # Dependencies
└── README.md               # This file
 ```
### Results
A brief summary of the results shows that both the loss masking and ray filtering strategies outperform the baseline model in terms of artifact reduction and overall visual quality. The quantitative metrics in `metrics_comparison.csv` provide a detailed comparison of the performance of the three models.
## Key Results
| Strategy | Static L1 Error | PSNR (dB) | Improvement |
|----------|-----------------|-----------|-------------|
| Baseline | 0.1067 | 9.00 | — |
| Loss Masking | 0.0000 | 18.35 | 107x |
| Ray Filtering | 0.0000 | 9.04 | 107x |

## Description of the Obtained Results
The results of our experiments, including the trained models, rendered images, and performance metrics, are available in the results folder.
- Static L1 Error: 0.1067 → 0.0000 (107x reduction)
- PSNR: 9.00 → 18.35 dB (2x improvement)
- Training: 7000 iterations, 16 Gaussians (fixed)
Demonstrates 107x static artifact reduction using 2D semantic masks in 3D Gaussian Splatting training.

## References
- Kerbl et al. (2023). 3D Gaussian Splatting
- Repository: github.com/prinssalex/3dgs-dynamic-masking-47

[1](https://arxiv.org/html/2510.18101v1)
[2](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
[3](https://www.cvlibs.net/publications/Chen2024ECCVb.pdf)
[4](https://cs.nyu.edu/~apanda/assets/papers/iclr25.pdf)
[5](https://proceedings.neurips.cc/paper_files/paper/2024/file/dd51dbce305433cd60910dc5b0147be4-Paper-Conference.pdf)
[6](https://isprs-archives.copernicus.org/articles/XLVIII-1-W5-2025/185/2025/isprs-archives-XLVIII-1-W5-2025-185-2025.pdf)
[7](https://arxiv.org/html/2506.05965v1)
[8](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_MaskGaussian_Adaptive_3D_Gaussian_Representation_from_Probabilistic_Masks_CVPR_2025_paper.pdf)
[9](https://w-m.github.io/3dgs-compression-survey/)
[10](https://www.sctheblog.com/blog/gaussian-splatting/)
[11](https://learnopencv.com/3d-gaussian-splatting/)
[12](https://github.com/kwea123/gaussian_splatting_notes)
[13](https://www.reddit.com/r/GaussianSplatting/comments/1hvycly/explaining_rendering_in_gaussian_splatting/)
[14](https://shi-yan.github.io/how_to_render_a_single_gaussian_splat/)
[15](https://arxiv.org/html/2506.02751v1)
[16](https://en.wikipedia.org/wiki/Gaussian_splatting)
[17](https://arxiv.org/html/2510.02884)
[18](https://www.visgraf.impa.br/Data/RefBib/PS_PDF/tutorial-sib2025/tutorial-sib2025.pdf)
[19](https://ieeexplore.ieee.org/iel8/7083369/11215960/11235983.pdf)
[20](https://github.com/graphdeco-inria/gaussian-splatting)

---
| Dec 2025 | Hypothesis PROVEN
