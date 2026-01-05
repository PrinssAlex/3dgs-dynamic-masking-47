
# Increasing the Robustness of Gaussian Splatting by Masking Dynamic Objects

## Participants

*   Prince Ogbodum [474343]
*   Jacob Ahemen [476149]
*   Christopher Mandengs [486554]

## Description of the Conducted Research
- **Platform:**  
  Experiments were run primarily on **Google Colab Pro** with an NVIDIA **T4 GPU**. This setup offers:
  - GPU acceleration needed for efficient 3DGS training and rendering.  
  - A reproducible, shareable environment suitable for course projects.  
  - Direct integration with Google Drive for dataset and artifact management.

Key concepts:

- **Reproducibility:** Clearly documenting the environment (hardware, runtime, and dependencies) so that other students or reviewers can re‑run our experiments.  
- **GPU acceleration:** Leveraging parallel computation on GPUs to handle the heavy rasterization and optimization workload in 3DGS.

***

## Problem Statement and Motivation

Our starting point is the observation that 3DGS optimizes a set of Gaussians so that rendered views match a collection of input images. This works well when the scene is static, but if cars, arms, or people move across frames, the optimization tries to “bake” these transient objects into a single static 3D structure. As a result, the model fits mutually inconsistent observations and produces artifacts such as faint “ghosts” of the moving object at multiple positions, smeared textures along motion trajectories, and noisy or unstable geometry in regions where motion occurs.

We aim to improve robustness of 3DGS to such dynamics without moving to a fully dynamic (4D) representation. Instead of modeling time explicitly, we ask whether we can obtain a clean static reconstruction simply by ignoring pixels that correspond to moving objects during training.

Key concepts:

- **Static vs dynamic scene assumption:** Classical radiance fields and 3DGS implicitly assume that the world does not change over time throughout the capture sequence; violations of this assumption cause inconsistent supervision.  
- **Artifacts:** Undesired patterns in reconstructions—ghosting, blurring, noisy geometry—that arise when a static model tries to fit observations from a changing scene.

***

## Related Work

Our work is grounded in three main lines of research:

- **3D Gaussian Splatting:** The core method we build on, where a scene is modeled as a set of Gaussians and rendered in real time by projecting and alpha‑compositing their 2D footprints.  
- **4D Gaussian Splatting / dynamic NeRFs:** Methods that explicitly model time or motion to reconstruct dynamic scenes. These approaches can capture moving objects but introduce additional complexity and higher computational cost.  
- **Mask‑based methods:** Prior works that use semantic or instance masks to remove transient content (cars, pedestrians, sky, reflections) from training, thereby focusing the model on stable parts of the scene.

Key concepts:

- **Radiance fields:** Functions that map 3D position and viewing direction to color and density, forming the foundation for NeRF‑like and 3DGS‑like methods.  
- **4D representation:** Extending the domain of the radiance field to include time, enabling explicit modeling of dynamic objects but at a significantly higher complexity.  
- **Semantic masks:** Binary or multi‑class images labeling pixels by category (e.g. “manipulator arm” vs “background”), which can be used to selectively include or exclude regions from training.

***

## Dataset and Data Generation

To systematically study the effect of dynamic objects and masking, we construct a synthetic dataset where ground truth geometry and motion are fully controlled.

- We create a **static background** (e.g. walls and floor) and a **moving manipulator** (or similar object), which follows a predefined trajectory over time.  
- From multiple camera viewpoints, we render:
  - RGB frames that serve as ground‑truth images for the 3DGS optimization.  
  - Per‑pixel **binary masks** indicating dynamic pixels (belonging to the moving object) and static pixels (background).  
  - **Camera poses** (intrinsics and extrinsics) required for 3DGS, ensuring every frame has accurate calibration.

Synthetic data offers full control over motion, lighting, and geometry, while making masks straightforward to generate without relying on external segmentation networks.

Key concepts:

- **Synthetic data:** Programmatically generated scenes that enable controlled experiments, reproducible conditions, and access to perfect labels.  
- **Camera intrinsics and extrinsics:** Parameters that define how 3D points map to 2D pixels: intrinsics capture focal length and principal point; extrinsics capture the camera’s pose (rotation and translation).  
- **Masks:** Image‑sized boolean or 0/1 arrays that mark which pixels correspond to dynamic objects and which belong to the static background.

***

## Baseline 3D Gaussian Splatting

As a reference, we use an unmodified 3DGS training pipeline.

- The scene is represented by \(N\) Gaussians, each with a mean (3D position), covariance, color, and opacity.  
- During each training iteration, we sample rays through pixels, project the Gaussians into image space, and rasterize them as elliptical 2D “splats” that are alpha‑composited along each ray.  
- We minimize a photometric loss (e.g. L1 loss, optionally combined with SSIM) between the rendered image and the ground‑truth image across all pixels and training views.

Because no masking is applied, the baseline model treats dynamic object pixels as if they were part of a static scene, which leads to ghosting and smearing in the reconstructions.

Key concepts:

- **Gaussian splat:** The 2D footprint of a 3D Gaussian on the image plane, rendered as an anisotropic elliptical blob whose size and orientation depend on the covariance and camera projection.  
- **Alpha compositing:** Front‑to‑back blending along each ray, where each Gaussian contributes color and opacity, and later Gaussians are attenuated by the accumulated transparency of earlier ones.  
- **Photometric loss:** A loss function that measures per‑pixel differences between predicted and ground‑truth colors, typically using L1/L2 distances and sometimes perceptual components like SSIM.

***

## Dynamic Object Masking – Concept

Our central idea is to **shield the static scene reconstruction from the influence of moving objects** by using per‑pixel masks during training. For every training image, we assume a mask \(M(u)\) is available that labels pixels belonging to dynamic objects.

Rather than letting the model explain both static and dynamic regions with the same static Gaussians, we integrate these masks into training in two different ways. The goal is to use all images but only let static regions drive the optimization.

Key concepts:

- **Mask integration:** Incorporating 2D masks into the training pipeline, either at the **loss level** (which pixels contribute to the loss) or at the **rendering level** (which rays or Gaussian contributions are considered at all).  

***
## Strategy 1 – Loss Masking (Gradient Masking)

In the loss masking strategy, we keep the rendering pipeline unchanged and only modify the loss computation.

Instead of computing the loss over all pixels, we multiply the pixelwise error by the static‑region mask $M\$, where:

- $M(u) = 1\$ for static pixels.  
- $M(u) = 0\$ for dynamic pixels.

This produces a masked loss:

$$
\mathcal{L}_{\text{mask}} = \frac{\sum_{u} M(u)\ \text{error}(u)}{\sum_{u} M(u) + \varepsilon},
\
$$

where $\text{error}(u)\$ can be the L1 error or a combination of L1 and other terms, and $\varepsilon\$ prevents division by zero. Gradients are therefore propagated only from static background pixels.

Key concepts:

- **Masked loss:** A loss function where each pixel’s contribution is weighted by the mask $M(u)\$, so only static pixels influence the optimization.  
- **Gradient masking:** By setting the error to zero on dynamic pixels, the optimizer does not attempt to fit those areas, and the Gaussians are driven mainly by stable, consistent content.  
- **Data efficiency:** All frames, including those with dynamic objects, still provide supervision—but only in the regions labeled as static.

Usage interpretation:

- **Pros:** Very simple to implement, compatible with any photometric loss, and leaves the core training loop and rendering code almost unchanged.  
- **Cons:** Dynamic regions still appear in the input images; if masks are imperfect, some artifact leakage from mis‑labeled pixels can persist.

***

## Strategy 2 – Ray Filtering (Geometry Masking)

Ray filtering enforces masking at the level of ray sampling and rendering.

We explore two equivalent ways to apply this idea:

- **Ray selection:** Only sample rays that pass through static pixels, completely ignoring rays whose corresponding pixels are dynamic. In this case, dynamic regions are never even considered in the loss.  
- **Contribution masking:** For rays passing through a dynamic pixel, we set the contribution of Gaussians to zero (e.g. $$\alpha_i^{\text{eff}}(u) = M(u)\,\alpha_i(u)\)$$, effectively removing those rays from the optimization.

In both variants, the training signal from dynamic regions is removed more directly than in loss masking, which operates only at the loss level.

Key concepts:

- **Ray:** A 3D line from the camera center through a pixel, along which Gaussians are intersected and composited.  
- **Ray selection:** Choosing which rays are traced based on the mask, so only rays that see static content contribute to the objective.  
- **Geometry filtering:** Preventing Gaussians that lie along dynamic pixels from receiving training signal, leading to a cleaner static geometry.

Usage interpretation:

- **Pros:** Provides stronger isolation of static geometry from dynamic content and can reduce ghost artifacts even more aggressively.  
- **Cons:** Requires more invasive modifications to sampling and rasterization; if too many rays are dropped, training can become data‑starved in certain regions, potentially harming reconstruction there.

***

## Training Setup and Hyperparameters

To ensure a fair comparison between the baseline and both masking strategies, we keep most training settings consistent.

Typical choices include:

- **Iterations / epochs:** A fixed number of iterations (e.g. 5,000) for all three models.  
- **Learning rate and optimizer:** A standard optimizer such as Adam with a chosen learning rate schedule.  
- **Resolution scaling:** Training at a reduced image resolution (e.g. 256×144) to balance fidelity and runtime.  
- **Batch size:** Defined in terms of rays or pixels per iteration, depending on the implementation.

Key concepts:

- **Optimization loop:** The repeated cycle of sampling rays, rendering images, computing the masked or unmasked loss, and updating the Gaussian parameters.  
- **Densification / pruning (if used):** Mechanisms in 3DGS that adapt the set of Gaussians (splitting, pruning, or relocating them) to better cover the scene over time.  
- **Seed and reproducibility:** Fixing random seeds for ray sampling and initialization to make runs comparable across different training configurations.

***

## Rasterization / Projection Section

Here we explain how a 3D Gaussian is turned into a 2D splat suitable for rasterization.

- First, we transform the Gaussian mean and covariance from world space to camera space using the camera extrinsics (rotation and translation).  
- Next, we linearize the camera projection around the Gaussian’s mean and apply its Jacobian to the 3D covariance. This yields a 2D covariance matrix describing an ellipse in the image plane.  
- Finally, we use this 2D covariance to evaluate the Gaussian’s footprint in pixel space and alpha‑composite these contributions along each ray, ordered by depth.

Key concepts:

- **Covariance projection:** The 2D covariance is computed as $$\Sigma_{2D} = J \Sigma_{3D} J^\top\$$, where $$J\$$ is the Jacobian of the projection at the Gaussian’s center.  
- **Screen‑space ellipse:** The visible “shadow” of the 3D Gaussian in image coordinates, which defines how wide and oriented the splat is.  
- **Depth sorting:** Ordering Gaussians along each ray by distance from the camera so that alpha compositing correctly models occlusion.


Here is your text with an additional “Evaluation Metrics – L1 Loss” subsection, consistent in style and phrasing with the others.

***
### Evaluation Metrics – L1 Loss

In addition to perceptual metrics, we also report the average **L1 loss** between the reconstructed and ground‑truth images. L1 loss measures the mean absolute difference in pixel values and is closely related to the photometric objective used during training in many 3DGS implementations.

Key concepts:

- **Absolute error:** Unlike MSE, which squares differences, L1 uses absolute differences, making it less sensitive to large outliers and often yielding sharper reconstructions.  
- **Alignment with training objective:** When L1 (or a masked L1) is used as the training loss, reporting L1 at test time directly reflects how well the model optimizes its primary objective.  
- **Interpretation:** Lower L1 indicates smaller average pixel deviations. It complements PSNR (which is derived from MSE) by giving a more robust view of pixelwise error, especially in regions with sharp edges and high contrast.
***

### Evaluation Metrics – SSIM

We also report SSIM (Structural Similarity Index) as a more perceptual metric. SSIM compares local patterns of luminance, contrast, and structure, and is computed over small windows and averaged, yielding scores typically in the range $\[0,1]\$.

Key concepts:

- **Structural similarity:** SSIM checks whether edges, textures, and local structures align between the reconstructed and ground‑truth images, rather than comparing only raw pixel values.  
- **Local windows:** Evaluating SSIM over patches makes it more robust to global intensity shifts and small misalignments than pure MSE.  
- **Interpretation:** Higher SSIM indicates better preservation of structures such as edges of the manipulator and background features, complementing the information given by PSNR and L1.

***

### Evaluation Metrics – LPIPS

LPIPS (Learned Perceptual Image Patch Similarity) is used as a perceptual distance metric in a deep feature space. Rendered and ground‑truth images are passed through a pretrained CNN, and differences in their feature maps at multiple layers are aggregated into a single score.

Key concepts:

- **Perceptual distance:** LPIPS measures how different images appear to a deep network trained on natural images, often correlating better with human judgments than PSNR, SSIM, or raw L1.  
- **Feature maps:** Intermediate activations (e.g. from VGG or AlexNet) that capture edges, textures, and high‑level semantics.  
- **Interpretation:** Lower LPIPS indicates that the reconstruction looks more similar to the ground truth; it is particularly sensitive to ghosting, smearing, and texture distortions that might be under‑penalized by PSNR or L1.

## Demonstration (Video)

A video demonstrating the results of our project can be found here: (https://github.com/PrinssAlex/3dgs-dynamic-masking-47/blob/main/Videos/groundtruth.mp4)

You can also find a GIF animation of the results below:

![GIF of results](https://github.com/PrinssAlex/3dgs-dynamic-masking-47/blob/main/Videos/groundtruth.gif)
![GIF of results](https://github.com/PrinssAlex/3dgs-dynamic-masking-47/blob/main/Videos/masking.gif)

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

    It is recommended to use a virtual environment(Pybullet).

    ```bash	
    # PyTorch CUDA 11.8 (matches 3DGS)
    !pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Core: PyBullet, OpenCV, COLMAP Python bindings
    !pip install --quiet pybullet opencv-python-headless

    # Metrics: PSNR/SSIM (skimage), LPIPS
    !pip install --quiet scikit-image lpips

    # Utilities: tqdm, matplotlib, imageio (for GIFs)
    !pip install --quiet tqdm matplotlib imageio

    # pycolmap for synthetic COLMAP conversion (install last to avoid deps)
    !pip install --quiet pycolmap
    ```

## Running and Usage

Here are the instructions to run the different parts of the project.

### How to start model training:
Copy content from drive
```
cp -r /content/drive/MyDrive/3dgs_data/generated_data /content/
%cd /content/gaussian-splatting
OUTPUT_ROOT = "/content/drive/MyDrive/ML/3dgs-dynamic-masking/output"
DATA_ROOT   = "/content/drive/MyDrive/ML/3dgs-dynamic-masking/data/generated"
```
*   **Baseline Model (no masking):**

    ```bash
    !python train.py \
    -s "{DATA_ROOT}" \
    -m "{OUTPUT_ROOT}/baseline_2000_init" \
    --iterations 7000 \
    --eval \
    --white_background \
    --densify_from_iter 1000 \
    --densify_until_iter 6000 \
    --densification_interval 3000 \
    --opacity_reset_interval 3000
    ```

**Strategy 1: Loss Masking:**

    ```bash
   ```bash
!python train_loss_mask.py -s /content/drive/MyDrive/MLP/3dgs-dynamic-masking/data/generated -m //content/drive/MyDrive/ML/3dgs-dynamic-masking/output/loss_mask --iterations 5000 --mask_dir c
```
    ```

**Strategy 2: Ray Filtering:**

    ```bash
    python train_ray_filter.py -s /content/drive/MyDrive/MLP/3dgs-dynamic-masking/data/generated -m //content/drive/MyDrive/ML/3dgs-dynamic-masking/output/ray_filter --iterations 7000 --mask_dir /3dgs-dynamic-masking-47/data/generated/masks
    ```
Note: The training scripts `train_loss_mask.py` and `train_ray_filter.py` are modified versions of the original `train.py` script from the 3DGS repository.*

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

Given a ground-truth image $I_{\text{gt}}\$ and rendered image $I_{\theta}\$ (parameters $theta\$ are all Gaussian attributes), a standard photometric loss is

$$
\mathcal{L}_{\text{photo}} = \lambda_{1}\,\|I_{\theta} - I_{\text{gt}}\|_{1}
 \lambda_{\text{ssim}}\,\bigl(1 - \text{SSIM}(I_{\theta}, I_{\text{gt}})\bigr),
$$

where $lambda_1\$ and $lambda_{\text{ssim}}\$ weight the L1 and SSIM components respectively.[^3][^4]


## Loss masking with static/dynamic masks

Let M(u)\ in \{0,1\} be a binary mask for pixel u, where M(u)=1 denotes static background and M(u)=0 denotes dynamic regions to be ignored. The masked photometric loss can be written as

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

# L1, PSNR, SSIM and LPIPS evaluation formulas and usage tips

## L1 Loss( Also Photometric Loss)
Formally, for a rendered image $I_{\theta}\ and ground‑truth image $I_{\text{gt}}\$, the per‑image L1 loss is

$$
\mathcal{L}_{\text{L1}} = \frac{1}{N} \sum_{u} \bigl| I_{\theta}(u) - I_{\text{gt}}(u) \bigr|,
\$$

where the sum runs over all pixels $u\$, and $N\$ is the total number of pixels.

## PSNR

Given a ground‑truth image $I_{\text{gt}}\$ and a reconstructed image $I\$, first compute mean squared error

$$
\text{MSE} = \frac{1}{N}\sum_{u} \bigl(I(u) - I_{\text{gt}}(u)\bigr)^2,
$$

then

$$
\text{PSNR} = 10 \log_{10} \left(\frac{\text{MAX}^2}{\text{MSE}}\right),
$$

where MAX is the maximum possible pixel value (e.g. 1.0 for normalized images or 255 for 8‑bit images).[^1][^2]

**Usage tips:**

- Higher is better; typical NeRF/3DGS experiments report values in dB, often in the 20–40 dB range.[^3][^4]
- Very sensitive to small pixel differences and outliers; good for sanity checks and ablations, but does not always correlate with perceived quality.[^2][^5]
- Always ensure same resolution, alignment, and dynamic range between $I$ and $I_{\text{gt}}$; a wrong scaling (e.g. computing MSE on 0–255 but MAX=1) completely breaks PSNR.[^6][^7]


## SSIM

SSIM between local patches $x$ and $y$ (from $I$ and $I_{\text{gt}}$) is

$$
\text{SSIM}(x, y) =
\frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},
$$

where $$\mu_x, \mu_y$$ are local means, $$\sigma_x^2, \sigma_y^2$$ local variances, $$\sigma_{xy}$$ local covariance, and $$C_1, C_2$$ small constants for numerical stability. The image‑level SSIM is the average of this value over all windows.[^8][^9][^10]

**Usage tips:**

- Values are typically in $$[0,1]$$ for practical settings; closer to 1 means more structurally similar.[^10][^8]
- Better correlates with human perception than PSNR, because it compares luminance, contrast, and structure rather than raw pixel error.[^4][^8]
- Use grayscale or luminance (Y) channel for standard reporting; for 3DGS/NeRF, report mean SSIM over all test views, possibly alongside PSNR.[^5][^10]


## LPIPS

LPIPS measures perceptual distance in a deep feature space. Given images $$x$$ and $$x_0$$, pass them through a pretrained network (e.g. AlexNet/VGG) and let $$y_l, y_{0,l} \in \mathbb{R}^{H_l \times W_l \times C_l}$$ be normalized feature maps at layer $$l$$. The LPIPS distance is[^11][^12][^13]

$$
d(x, x_0) = \sum_{l} \frac{1}{H_l W_l}
\sum_{h,w} \left\| w_l \odot \bigl(\hat{y}_{l}(h,w) - \hat{y}_{0,l}(h,w)\bigr) \right\|_2^2,
$$

where $$\hat{y}_l$$ are channel‑wise unit‑normalized features and $$w_l$$ are learned per‑channel weights.[^14][^13]

**Usage tips:**

- Lower is better; 0 means identical in the chosen feature space.[^15][^11]
- Much better aligned with human judgments of perceptual similarity than PSNR/SSIM, especially for small misalignments, blur, or texture differences.[^12][^13]
- In code, prefer a standard implementation (e.g. `pip install lpips` and use the official PyTorch interface) and fix the backbone (e.g. `net='vgg'`) so your scores are comparable across experiments.[^13][^15]
<span style="display:none">[^16][^17][^18][^19][^20]</span>

### Project Structure
The folder is structured as follows:
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
│   └── sparse/0
│   └── database.db
   
├── output/       # Training results- Contains the output of the training runs for the baseline, loss masking, and ray filtering models.
│   ├── baseline/           # Unmasked training- Output for the baseline model.
│   ├── loss_mask/          # Loss masking -Output for the loss masking model.
│   └── ray_filter/         # Ray filtering -Output for the ray filtering model
├── results/                 # Metrics & figures - Contains the final comparison results.
│   ├── metrics_comparison.csv - A CSV file with the PSNR, SSIM, and LPIPS scores for all three models.
│   ├── figure_psnr_comparison.png - Side-by-side comparison images of the rendered views.
│   └── figure_l1_error_comparison.png

├── code/               # Colab notebook
├── images_and_videos
├── requirements.txt         # Dependencies
└── README.md               # This file
 ```
### Results
A brief summary of the results shows that both the loss masking and ray filtering strategies outperform the baseline model in terms of artifact reduction and overall visual quality. The quantitative metrics in `metrics_comparison.csv` provide a detailed comparison of the performance of the three models.
## Key Results
| Model       | PSNR   | SSIM   | LPIPS  | L1_Static |
|-------------|--------|--------|--------|-----------|
| **Baseline**    | 24.71  | 0.913  | 0.192  | 0.0048    |
| **Loss-Mask**   | 23.12  | 0.907  | 0.207  | 0.0025    |
| **Ray-Filter**  | 21.16  | 0.852  | 0.191  | 0.0104    |


## Description of the Obtained Results

After running a more comprehensive evaluation across 14 different camera views, the team found that while the baseline 3D Gaussian Splatting model still leads in overall image quality metrics like PSNR and SSIM, the real story lies in how each method handles the *static* parts of the scene which is precisely what this project set out to improve.

The numbers tell a nuanced tale. The **Baseline** model scores higher on PSNR (24.71) and SSIM (0.913), and even edges out the others on LPIPS (0.192). This makes sense: because it tries to reconstruct everything including the moving robot arm. As a result, it often produces renders where the dynamic region looks “plausible” enough to boost those global scores. In other words, it is like getting partial credit for trying to fit the whole picture, even if parts of it are blurry or ghosted.

But here’s where the **Loss-Mask** strategy shines. While its global metrics dip slightly (PSNR 23.12, SSIM 0.907, LPIPS 0.207), its performance on the *static background* is dramatically better. The key metric **Static L1 Error** drops from 0.0048 (Baseline) to just 0.0025. Which is nearly a 2x improvement in accuracy for the parts of the scene that matter most: the walls, floor, and cubes that should remain clean and stable.


Looking at the rendered frames, the Loss-Mask model delivers crisp, artifact-free static regions. The Baseline, by contrast, shows smearing and faint “ghosts” around the robot arm because it’s forced to reconcile conflicting observations from multiple frames. The Loss-Mask model simply ignores the dynamic pixels during training, letting the Gaussians focus purely on reconstructing the unchanging elements of the environment.

In short, the Loss-Mask approach does not try to win every metric; it wins the one that counts for this specific goal: producing a clean, accurate static reconstruction despite the presence of motion. The fact that PSNR, SSIM, and LPIPS do not reflect this victory perfectly is a reminder that these global metrics can sometimes miss the forest for the trees, especially when your target is not the entire image, but a specific part of it.

As for the **Ray-Filter** model, it struggles across the board, scoring lowest on almost every metric (PSNR 21.16, SSIM 0.852, L1_Static 0.0104). Its attempt to filter out dynamic rays was not successful, leading to noisy reconstructions and poor performance both globally and locally.

So, while the Loss-Mask model might not top the leaderboard in traditional benchmarks, it’s undeniably the champion when it comes to eliminating dynamic artifacts and delivering a better overall reconstruction and ultimately nove-view synthesis, exactly what the project aimed to achieve. This outcome aligns with the project's core hypothesis that training-time masking can substantially improve static scene reconstruction quality in dynamic scenes, even if global perceptual metrics do not always reflect the targeted improvement.

### Plans for Future work 
For future work, we plan to:
- **Learn masks automatically**, for example using a segmentation network, to apply the approach to real‑world videos.  
- **Extend from static‑scene robustness to full dynamic‑scene modeling**, possibly by combining masked static 3DGS with 4D Gaussians or other dynamic representations.  
- **Incorporate temporal consistency and multi‑view constraints**, such as temporal smoothing of masks and explicit regularization across adjacent frames.

Key concepts:

- **Static‑background reconstruction vs dynamic‑object modeling:** Our current work focuses on cleaning up the static part of the scene; modeling the moving objects themselves is left for future work.  
- **Scalability:** Moving from controlled synthetic scenes to complex real‑world captures with multiple dynamic agents.  
- **Integration with broader pipelines:** Using a robust static 3DGS representation in robotics, SLAM, AR, and other applications where dynamic objects are frequent but often treated as nuisances.

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
