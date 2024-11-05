

<div align="center">
    <h2> <img width="30" alt="pipeline" src="assets/logo.png"><a href="https://mvpaint.github.io"><span style="color: #FF69B4;">MVP</span><span style="color: #1E90FF;">aint</span>: Synchronized Multi-View Diffusion for Painting Anything 3D</a></h2>

<p align="center">
  <a href="https://mvpaint.github.io/">Project Page</a> •
  <a href="https://arxiv.org/abs/2411.02336">Arxiv</a> •
  <a href="#citation">Citation
</p>

</div>

## MVPaint

<div align="center">
<img width="720" alt="pipeline" src="assets/teaser-480p.gif">
<p><b>MVPaint</b> generates <b>multi-view consistent</b> textures with <b>arbitrary UV unwrapping</b> and <b>high generation versatility</b>.</p>
</div>

<details>
<summary><b>Introducing MVPaint</b></summary>
    <br></br>
    <div align="center">
    <img width="720" alt="pipeline" src="assets/pipeline.jpg">
    </div>
    <br></br>
    Texturing is a crucial step in the 3D asset production workflow, which enhances the visual appeal and diversity of 3D assets. Despite recent advancements in generation-based texturing, existing methods often yield subpar results, primarily due to local discontinuities, inconsistencies across multiple views, and their heavy dependence on UV unwrapping outcomes. To tackle these challenges, we propose a novel generation-refinement 3D texturing framework called <b>MVPaint</b>, which can generate high-resolution, seamless textures while emphasizing multi-view consistency. MVPaint mainly consists of three key modules. <b>1) Synchronized Multi-view Generation (SMG).</b> Given a 3D mesh model, MVPaint first simultaneously generates multi-view images by employing a SMG model, which leads to coarse texturing results with unpainted parts due to missing observations. <b>2) Spatial-aware 3D Inpainting (S3I).</b> To ensure complete 3D texturing, we introduce the S3I method, specifically designed to effectively texture previously unobserved areas. <b>3) UV Refinement (UVR).</b> Furthermore, MVPaint employs a UVR module to improve the texture quality in the UV space, which first performs a UV-space Super-Resolution, followed by a Spatial-aware Seam-Smoothing algorithm for revising spatial texturing discontinuities caused by UV unwrapping. Extensive experimental results demonstrate that MVPaint surpasses existing state-of-the-art methods. Notably, MVPaint could generate high-fidelity textures with minimal Janus issues and highly enhanced cross-view consistency.

</details>



## News

- [2024/10/31] Upload paper and init project.


## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@article{cheng2024mvpaint,
  title={MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D}, 
  author={Wei Cheng and Juncheng Mu and Xianfang Zeng and Xin Chen and Anqi Pang and Chi Zhang and Zhibin Wang and Bin Fu and Gang Yu and Ziwei Liu and Liang Pan},
  journal={arXiv preprint arxiv:2411.02336},
  year={2024}
}
```

## Acknowledgments

Thanks to these amazing works which MVPaint is built upon: [MVDream](https://github.com/bytedance/MVDream), [SyncMVD](https://github.com/LIU-Yuxin/SyncMVD) and [Paint3D](https://github.com/OpenTexture/Paint3D)
