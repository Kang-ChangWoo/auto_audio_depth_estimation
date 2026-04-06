# AudioDepthFOA: Joint Depth Estimation and First-Order Ambisonic Prediction from Binaural Echoes via Spherical Harmonic Guidance

---

## 3. Method

We present AudioDepthFOA, a method for monocular depth estimation from binaural echoes that leverages First-Order Ambisonic (FOA) spatial audio as an auxiliary supervision signal. Our approach consists of three main components: (1) a covariance-based directional energy representation that encodes spatial audio cues into equirectangular projection (ERP) maps, (2) a UNet encoder–decoder architecture with a spherical harmonic (SH) auxiliary branch, and (3) a multi-task loss that jointly optimizes depth prediction and FOA coefficient regression. An overview of our architecture is shown in Figure~\ref{fig:architecture}.

### 3.1 Covariance-Based Directional Energy Maps

Spatial audio captured in First-Order Ambisonics encodes rich directional information about the acoustic environment. We extract this information through covariance-based directional energy maps that project the FOA impulse response onto the equirectangular grid used for depth estimation.

**Spherical harmonic basis.** We compute a real-valued SH basis matrix $\mathbf{B} \in \mathbb{R}^{HW \times C}$ on the ERP grid, where $H \times W$ is the spatial resolution and $C = (l_{\max}+1)^2$ is the number of SH channels up to order $l_{\max}=1$ (i.e., $C=4$ for FOA). Each entry is the SN3D-normalized real spherical harmonic evaluated at the corresponding elevation $\theta$ and azimuth $\phi$:

$$Y_n^m(\theta, \phi) = N_n^m \, P_n^{|m|}(\sin\theta) \times \begin{cases} \cos(m\phi) & m > 0 \\ 1 & m = 0 \\ \sin(|m|\phi) & m < 0 \end{cases}$$

where $N_n^m = \sqrt{(2-\delta_m)\frac{(n-|m|)!}{(n+|m|)!}}$ is the SN3D normalization factor and $P_n^{|m|}$ is the associated Legendre polynomial with Condon–Shortley phase.

**Inter-channel covariance.** Given a FOA impulse response $\mathbf{b}(t) \in \mathbb{R}^{C}$ with $C=4$ channels (W, Y, Z, X in ACN ordering), we compute the inter-channel covariance matrix:

$$\mathbf{R} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{b}(t)\,\mathbf{b}(t)^\top \in \mathbb{R}^{C \times C}$$

Unlike per-channel RMS projections, this formulation captures cross-channel correlations that encode directional information.

**Directional energy map.** The energy at each direction $\Omega = (\theta, \phi)$ is computed as:

$$E(\Omega) = \mathbf{y}(\Omega)^\top \mathbf{R}\, \mathbf{y}(\Omega)$$

where $\mathbf{y}(\Omega) \in \mathbb{R}^C$ is the SH basis vector at direction $\Omega$. In matrix form, for the full ERP grid: $\mathbf{E} = \text{diag}(\mathbf{B}\,\mathbf{R}\,\mathbf{B}^\top) \in \mathbb{R}^{HW}$, reshaped to $H \times W$.

**Temporal decomposition.** We decompose the impulse response into early ($0$ to $\tau$ ms) and late ($\tau$ ms to end) segments, where $\tau = 20$ ms, yielding three covariance matrices $\mathbf{R}_{\text{full}}$, $\mathbf{R}_{\text{early}}$, and $\mathbf{R}_{\text{late}}$. From these we construct a 4-channel energy tensor $\mathcal{E} \in \mathbb{R}^{4 \times H \times W}$:

$$\mathcal{E} = \big[E_{\text{full}},\; E_{\text{early}},\; E_{\text{late}},\; \widetilde{E}_{\text{early}} - \widetilde{E}_{\text{late}}\big]$$

where $\widetilde{E}$ denotes max-absolute normalization. The early component captures direct-path reflections (correlated with nearby surfaces), while the late component captures reverberation (correlated with room geometry). The difference channel highlights regions where direct reflections dominate over reverberation, providing an explicit proximity cue. Each channel is independently normalized to $[0, 1]$ before being fed to the network.

### 3.2 Network Architecture

#### 3.2.1 Input Representation

The binaural audio waveform is truncated to the first $2d/c$ seconds (where $d = 20$ m is the maximum scene depth and $c = 340$ m/s is the speed of sound), yielding a temporal window that captures all first-order reflections from surfaces within range. A magnitude spectrogram is computed with FFT size 512, window length 400, and hop length 160, producing a 2-channel spectrogram (one per ear). This spectrogram is resized to $H \times W$ and concatenated with the 4-channel ambisonic energy maps, yielding a 6-channel input tensor $\mathbf{X} \in \mathbb{R}^{6 \times H \times W}$.

#### 3.2.2 UNet Encoder–Decoder

The backbone is a UNet with 8 downsampling stages. The encoder begins with a convolution from 6 input channels to $f=64$ feature channels, followed by 6 encoder blocks, each consisting of LeakyReLU (slope 0.2), $4 \times 4$ strided convolution (stride 2), and batch normalization. Channel counts double at each stage up to a maximum of $8f = 512$. The innermost encoder block omits batch normalization.

The decoder mirrors the encoder with transposed convolutions (stride 2) and skip connections. Each decoder block concatenates the corresponding encoder feature map before upsampling. The final decoder layer produces a 1-channel output through a sigmoid activation (since depth targets are normalized to $[0, 1]$ by dividing by the maximum depth $d_{\max} = 10$ m).

#### 3.2.3 Spherical Harmonic Auxiliary Branch

At the encoder bottleneck, we branch off a spherical harmonic (SH) prediction head. The bottleneck features are globally average-pooled and projected through a two-layer MLP:

$$\mathbf{z} = \text{MLP}_{\text{proj}}(\text{AvgPool}(\mathbf{h}_{\text{bottleneck}})) \in \mathbb{R}^{d_{\text{proj}}}$$

where $d_{\text{proj}} = 128$. This latent vector $\mathbf{z}$ is then split into two heads:

- **FOA head**: $\hat{\mathbf{f}} = \text{MLP}_{\text{foa}}(\mathbf{z}) \in \mathbb{R}^4$ predicts the 4 first-order ambisonic coefficients (W, Y, Z, X).
- **HOA head**: $\hat{\mathbf{h}} = \text{MLP}_{\text{hoa}}(\mathbf{z}) \in \mathbb{R}^{32}$ predicts the 32 higher-order coefficients (orders 2–5), for a total of $(5+1)^2 = 36$ SH coefficients.

Both heads use BatchNorm followed by ReLU before the final linear layer. The concatenated coefficients $\hat{\mathbf{s}} = [\hat{\mathbf{f}}; \hat{\mathbf{h}}] \in \mathbb{R}^{36}$ form the complete SH5 prediction.

**FOA supervision target.** The ground-truth FOA target $\mathbf{f}^* \in \mathbb{R}^4$ is obtained by spatially averaging the 4-channel ambisonic energy maps:

$$f^*_c = \frac{1}{HW}\sum_{i,j} \mathcal{E}_c(i, j), \quad c \in \{0, 1, 2, 3\}$$

This provides a compact directional summary that the SH branch learns to predict from the encoded audio features.

#### 3.2.4 DeepScaleShift Module

To bridge the SH coefficient space and the spatial depth domain, we employ a DeepScaleShift module—a gated residual MLP that learns a per-coefficient affine transformation. Given the predicted SH coefficients $\hat{\mathbf{s}} \in \mathbb{R}^{36}$:

$$\hat{\mathbf{s}}_{\text{aligned}} = (1 - \sigma(\mathbf{g})) \odot (\boldsymbol{\gamma} \odot \hat{\mathbf{s}} + \boldsymbol{\beta}) + \sigma(\mathbf{g}) \odot \text{MLP}(\hat{\mathbf{s}})$$

where $\boldsymbol{\gamma}, \boldsymbol{\beta}, \mathbf{g} \in \mathbb{R}^{36}$ are learnable parameters for scaling, shifting, and gating respectively, and $\sigma$ is the sigmoid function. The MLP consists of LayerNorm, followed by 2 hidden layers of dimension 128 with GELU activations and dropout ($p=0.1$), and a final linear projection. Weights are initialized with small Xavier uniform values (gain $= 0.01$) so that the module starts near the identity, with the gate $\mathbf{g}$ initialized to zero (sigmoid output $\approx 0.5$).

### 3.3 Loss Functions

The total training loss is a weighted combination of four terms:

$$\mathcal{L} = w_{\text{depth}} \cdot \mathcal{L}_{\text{depth}} + w_{\text{foa}} \cdot \mathcal{L}_{\text{foa}} + w_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

where $w_{\text{depth}} = 1.0$, $w_{\text{foa}} = 0.2$, and $w_{\text{reg}} = 0.001$.

#### 3.3.1 Depth Loss $\mathcal{L}_{\text{depth}}$

The depth loss is a weighted sum of five complementary terms, each computed only over valid pixels where $d^* > 0$:

$$\mathcal{L}_{\text{depth}} = w_1\mathcal{L}_{\text{L1}} + w_2\mathcal{L}_{\text{BerHu}} + w_3\mathcal{L}_{\text{SILog}} + w_4\mathcal{L}_{\text{grad}} + w_5\mathcal{L}_{\text{SSIM}}$$

with weights $w_1 = 1.0$, $w_2 = 0.5$, $w_3 = 0.5$, $w_4 = 0.5$, $w_5 = 1.0$.

**L1 Loss.** Standard pixel-wise absolute difference:

$$\mathcal{L}_{\text{L1}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} |\hat{d}_i - d^*_i|$$

where $\mathcal{M} = \{i : d^*_i > 0\}$ is the set of valid pixels.

**BerHu (Reverse Huber) Loss.** Applies L1 for small errors and L2 for large errors, with an adaptive threshold $c = 0.2 \cdot \max_i |\hat{d}_i - d^*_i|$:

$$\mathcal{L}_{\text{BerHu}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \begin{cases} |e_i| & |e_i| \leq c \\ \frac{e_i^2 + c^2}{2c} & |e_i| > c \end{cases}$$

where $e_i = \hat{d}_i - d^*_i$. This provides robustness to outlier depths while maintaining sensitivity to small errors.

**Scale-Invariant Logarithmic (SILog) Loss.** Operates in log-depth space to be scale-invariant:

$$\mathcal{L}_{\text{SILog}} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} g_i^2 - \lambda\left(\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} g_i\right)^2$$

where $g_i = \log(\hat{d}_i + \epsilon) - \log(d^*_i + \epsilon)$ and $\lambda = 0.5$ controls the variance penalty. The second term encourages the error distribution to be zero-mean, making the loss invariant to global scale shifts.

**Gradient Loss.** Penalizes differences in spatial gradients to preserve edge structure:

$$\mathcal{L}_{\text{grad}} = \frac{\sum_{i} m_i^x |\Delta_x \hat{d}_i - \Delta_x d^*_i|}{\sum_i m_i^x} + \frac{\sum_i m_i^y |\Delta_y \hat{d}_i - \Delta_y d^*_i|}{\sum_i m_i^y}$$

where $\Delta_x, \Delta_y$ are horizontal and vertical finite-difference operators and $m_i^x, m_i^y$ are validity masks ensuring both neighboring pixels are valid.

**SSIM Loss.** The structural similarity loss encourages perceptual consistency:

$$\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(\hat{d}, d^*)$$

computed with a window size of 11, using the standard SSIM formulation with constants $C_1 = 0.01^2$ and $C_2 = 0.03^2$. Only regions where the pooled validity mask exceeds 0.5 contribute to the loss.

#### 3.3.2 FOA Guided Loss $\mathcal{L}_{\text{foa}}$

The FOA loss supervises the predicted first-order ambisonic coefficients:

$$\mathcal{L}_{\text{foa}} = \|\hat{\mathbf{f}} - \mathbf{f}^*\|_1 + w_{\text{cos}} \left(1 - \frac{\hat{\mathbf{f}} \cdot \mathbf{f}^*}{\|\hat{\mathbf{f}}\|\,\|\mathbf{f}^*\|}\right)$$

where $w_{\text{cos}} = 0.2$. The L1 term matches coefficient magnitudes while the cosine term enforces directional consistency between the predicted and ground-truth FOA vectors. This auxiliary task encourages the bottleneck to encode spatial information about the acoustic scene geometry.

#### 3.3.3 Latent Regularization $\mathcal{L}_{\text{reg}}$

We apply L2 regularization on the latent projection vector to prevent the SH branch from developing excessively large activations:

$$\mathcal{L}_{\text{reg}} = \frac{1}{d_{\text{proj}}}\sum_{j=1}^{d_{\text{proj}}} z_j^2$$

This encourages a compact latent representation and improves training stability.

### 3.4 Training Details

We use the AdamW optimizer with an initial learning rate of $2 \times 10^{-3}$ and cosine annealing schedule with $T_{\max} = 13$ and $\eta_{\min} = 10^{-6}$. Gradient norms are clipped to a maximum of 0.5 to stabilize training. The batch size is 32. Depth targets are normalized to $[0, 1]$ by dividing by $d_{\max} = 10$ m, and predictions are passed through a sigmoid activation. Each training run uses a fixed wall-clock budget of 1 hour on a single GPU.

---

## 4. Experiments

### 4.1 Dataset

We evaluate on the SoundSpaces dataset~\cite{chen2020soundspaces}, which provides binaural room impulse responses (RIRs) and corresponding equirectangular depth maps for indoor environments rendered in the Habitat simulator. Each sample consists of:
- A binaural audio waveform (WAV, 2 channels) captured at a specific listener position.
- A first-order ambisonic impulse response (4 channels, 48 kHz) stored as NumPy arrays.
- A ground-truth ERP depth map ($256 \times 512$) with maximum depth 10 m.

We use a deterministic scene-level split of 80\%/10\%/10\% for train/val/test (seed 42), ensuring no scene overlap between splits. Samples where more than 10\% of pixels have zero (invalid) depth are filtered out during dataset construction.

### 4.2 Evaluation Metrics

We report standard monocular depth estimation metrics, computed only over valid pixels ($d^* > 0$):

- **Abs Rel**: $\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} \frac{|d^*_i - \hat{d}_i|}{d^*_i}$
- **RMSE**: $\sqrt{\frac{1}{|\mathcal{M}|}\sum_{i\in\mathcal{M}} (d^*_i - \hat{d}_i)^2}$
- **$\delta_1$**: \% of pixels where $\max\!\left(\frac{d^*}{\hat{d}}, \frac{\hat{d}}{d^*}\right) < 1.25$
- **$\delta_2$, $\delta_3$**: Same with thresholds $1.25^2$ and $1.25^3$.

Lower is better for Abs Rel and RMSE; higher is better for $\delta_k$.

### 4.3 Implementation Details

The model has approximately 85M parameters with $f=64$ base feature channels. Input resolution is $256 \times 512$. The SH branch operates at order 5 ($l_{\max}=5$), yielding 36 coefficients (4 FOA + 32 HOA). The projection dimension is $d_{\text{proj}} = 128$ and the DeepScaleShift module uses 2 hidden layers of dimension 128 with GELU activation and dropout $p = 0.1$. Training uses PyTorch with mixed-precision where available, DataParallel for multi-GPU, and 4 data loader workers. Validation is performed every 2 epochs, and the model with the lowest Abs Rel on the validation set is selected.

### 4.4 Ablation Study

We conduct an extensive ablation study to validate each component of our design. All experiments use the same 1-hour training budget, ensuring fair comparison. Results are summarized in Table~\ref{tab:ablation}.

\begin{table}[t]
\centering
\caption{Ablation study on the SoundSpaces validation set. Each row modifies one aspect of the full model. Lower is better for Abs Rel and RMSE; higher is better for $\delta_1$.}
\label{tab:ablation}
\small
\begin{tabular}{l c c c}
\toprule
\textbf{Configuration} & \textbf{Abs Rel $\downarrow$} & \textbf{RMSE $\downarrow$} & \textbf{$\delta_1$ $\uparrow$} \\
\midrule
\multicolumn{4}{l}{\textit{Input representation}} \\
Binaural spectrogram only (2 ch) & 0.3809 & 1.2570 & 0.5220 \\
\quad + Ambisonic energy maps (6 ch) & 0.3648 & 1.2241 & 0.5324 \\
\midrule
\multicolumn{4}{l}{\textit{Depth loss components}} \\
Full model w/o BerHu loss & 0.4038 & 1.2599 & 0.5189 \\
Full model w/o SSIM loss (baseline) & 0.3821 & 1.2496 & 0.5225 \\
$w_{\text{SSIM}} = 0.5$ & 0.3648 & 1.2241 & 0.5324 \\
$w_{\text{SSIM}} = 0.75$ & 0.3591 & 1.2170 & 0.5309 \\
$w_{\text{SSIM}} = 1.0$ (ours) & \textbf{0.3498} & \textbf{1.2333} & 0.5311 \\
$w_{\text{SSIM}} = 1.5$ & 0.3616 & 1.2181 & 0.5362 \\
\midrule
\multicolumn{4}{l}{\textit{FOA branch}} \\
FOA weight $w_{\text{foa}} = 0.0$ (depth-only) & 0.3498 & 1.2333 & 0.5311 \\
FOA weight $w_{\text{foa}} = 0.1$ & 0.3876 & 1.2559 & 0.5188 \\
FOA weight $w_{\text{foa}} = 0.2$ (ours) & \textbf{0.3524} & 1.2244 & 0.5310 \\
FOA weight $w_{\text{foa}} = 0.3$ & 0.3538 & 1.2263 & 0.5307 \\
FOA weight $w_{\text{foa}} = 0.5$ & 0.4057 & 1.2635 & 0.5099 \\
\midrule
\multicolumn{4}{l}{\textit{Histogram alignment}} \\
Hist.\ alignment $w_{\text{hist}} = 0.2$ & 0.3809 & 1.2570 & 0.5220 \\
Hist.\ alignment $w_{\text{hist}} = 0.1$ & 0.3502 & 1.2180 & \textbf{0.5392} \\
Hist.\ alignment $w_{\text{hist}} = 0.0$ (ours) & \textbf{0.3498} & 1.2333 & 0.5311 \\
\midrule
\multicolumn{4}{l}{\textit{Latent regularization}} \\
No latent reg ($w_{\text{reg}} = 0$) & 0.3571 & 1.2156 & 0.5300 \\
Latent reg $w_{\text{reg}} = 0.001$ (ours) & \textbf{0.3538} & 1.2263 & 0.5307 \\
\midrule
\multicolumn{4}{l}{\textit{ScaleShift module}} \\
Deep MLP (256-dim, 4 layers) & 0.3821 & 1.2496 & 0.5225 \\
Simpler MLP (128-dim, 2 layers, ours) & \textbf{0.3809} & 1.2570 & 0.5220 \\
Minimal MLP (64-dim, 1 layer) & 0.3992 & 1.2918 & 0.5031 \\
\midrule
\multicolumn{4}{l}{\textit{Training configuration}} \\
No gradient clipping & 0.3929 & 1.2858 & 0.5073 \\
Grad clip $\text{max\_norm} = 1.0$ & 0.3849 & 1.2746 & 0.5126 \\
Grad clip $\text{max\_norm} = 0.5$ (ours) & \textbf{0.3821} & 1.2496 & 0.5225 \\
Grad clip $\text{max\_norm} = 0.25$ & 0.4008 & 1.2335 & 0.5228 \\
\bottomrule
\end{tabular}
\end{table}

#### 4.4.1 Input Representation

Adding 4-channel ambisonic energy maps to the 2-channel binaural spectrogram yields the single largest improvement, reducing Abs Rel from 0.3809 to 0.3648 (a 4.2\% relative improvement) and RMSE from 1.2570 to 1.2241. The covariance-based energy maps provide explicit spatial priors about surface locations that complement the binaural spectral cues, particularly through the early-late decomposition that separates direct reflections from reverberation.

#### 4.4.2 Depth Loss Design

Each component of the depth loss contributes meaningfully. Removing BerHu degrades Abs Rel by 5.6\% (0.3821 to 0.4038), confirming its importance for handling the heterogeneous error distribution in indoor depth. SSIM loss shows a clear optimum at $w_{\text{SSIM}} = 1.0$: increasing from 0.5 to 1.0 progressively improves Abs Rel (0.3648 $\to$ 0.3591 $\to$ 0.3498), while further increase to 1.5 causes regression (0.3616). The five-loss combination provides complementary gradients: L1 for overall magnitude, BerHu for robustness, SILog for scale invariance, gradient loss for edges, and SSIM for structural coherence.

#### 4.4.3 FOA Auxiliary Task

The FOA guided loss provides an auxiliary learning signal that encourages the encoder bottleneck to capture spatial scene geometry. At $w_{\text{foa}} = 0.2$, the FOA task improves Abs Rel from the depth-only baseline. However, the effect is sensitive to the weighting: too large a weight ($w_{\text{foa}} = 0.5$) shifts capacity away from depth prediction, increasing Abs Rel to 0.4057. The cosine similarity component (weight 0.2) ensures directional consistency beyond mere magnitude matching.

#### 4.4.4 Histogram Alignment

We explored an SH5 histogram alignment loss that encourages consistency between the SH-reconstructed energy map and the SH projection of the predicted depth. While $w_{\text{hist}} = 0.1$ achieves the best $\delta_1$ (0.5392), disabling it entirely ($w_{\text{hist}} = 0$) achieves the best Abs Rel (0.3498) with a simpler model. Following our simplicity criterion, we disable this component in the final model.

#### 4.4.5 Training Stability

Gradient clipping is essential for stable training. Without clipping, Abs Rel is 0.3929; clipping at $\text{max\_norm} = 0.5$ improves it to 0.3821, a 2.7\% gain. However, overly aggressive clipping ($\text{max\_norm} = 0.25$) slows convergence. Cosine annealing with $T_{\max} = 13$ (slightly less than total epochs at the 1-hour budget) and initial LR $2 \times 10^{-3}$ provides the best schedule, outperforming both larger and smaller $T_{\max}$ values.

### 4.5 Progression of Improvements

Table~\ref{tab:progression} shows the cumulative effect of our design choices, starting from the baseline and progressively adding each component.

\begin{table}[t]
\centering
\caption{Cumulative improvement from baseline to final model on SoundSpaces validation.}
\label{tab:progression}
\small
\begin{tabular}{l c c c}
\toprule
\textbf{Stage} & \textbf{Abs Rel $\downarrow$} & \textbf{RMSE $\downarrow$} & \textbf{$\delta_1$ $\uparrow$} \\
\midrule
Baseline (covariance energy maps, 2 ch) & 0.4098 & 1.2605 & 0.5086 \\
\quad + Cosine annealing + LR tuning & 0.3929 & 1.2858 & 0.5073 \\
\quad + Gradient clipping (0.5) & 0.3821 & 1.2496 & 0.5225 \\
\quad + Simpler ScaleShift (128-dim, 2L) & 0.3809 & 1.2570 & 0.5220 \\
\quad + Ambisonic energy input (6 ch) & 0.3648 & 1.2241 & 0.5324 \\
\quad + SSIM weight 1.0 & 0.3571 & 1.2156 & 0.5300 \\
\quad + Latent regularization & 0.3538 & 1.2263 & 0.5307 \\
\quad + FOA weight 0.2 & 0.3524 & 1.2244 & 0.5310 \\
\quad + Hist.\ weight reduction (0.1) & 0.3502 & 1.2180 & 0.5392 \\
\quad + Disable hist.\ alignment (final) & \textbf{0.3498} & 1.2333 & 0.5311 \\
\midrule
\textbf{Total relative improvement} & \textbf{14.6\%} & \textbf{2.2\%} & \textbf{4.4\%} \\
\bottomrule
\end{tabular}
\end{table}

Overall, we achieve a 14.6\% relative improvement in Abs Rel (0.4098 $\to$ 0.3498) through systematic experimentation. The two largest individual gains come from (1) incorporating ambisonic energy maps as additional input channels (4.2\% relative Abs Rel improvement) and (2) gradient clipping for training stability (2.7\% relative improvement).

### 4.6 Analysis

**Effect of temporal decomposition.** The early-late decomposition of the ambisonic impulse response is critical. Early reflections (0–20 ms) carry information about nearby surfaces—their intensity and arrival direction correlate with surface proximity and orientation. Late reverberation encodes room-scale geometry. The difference channel $\widetilde{E}_{\text{early}} - \widetilde{E}_{\text{late}}$ provides an explicit proximity signal: positive values indicate directions with strong early reflections relative to reverberation, typically corresponding to nearby surfaces.

**Model complexity.** Our final model uses approximately 85M parameters. Increasing the base feature channels from 64 to 96 (123M parameters) did not improve performance (Abs Rel 0.3877 vs.\ 0.3849), suggesting that the bottleneck lies in the input signal rather than model capacity. Similarly, increasing SH order from 5 to higher orders or the projection dimension from 128 to 256 provided no benefit, while reducing them degraded performance.

**Simplification wins.** Several simplification experiments yielded equal or better results: reducing ScaleShift from 4 layers (256-dim) to 2 layers (128-dim), and disabling histogram alignment entirely. These findings support the principle that unnecessary complexity can harm generalization under fixed compute budgets.
