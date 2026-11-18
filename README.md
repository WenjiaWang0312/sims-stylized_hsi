<h1 align="center">[ICCV2025] SIMS: Simulating Stylized Human-Scene Interactions<br>with Retrieval-Augmented Script Generation</h1>

<div align="center">
    <p>
        <a href="https://wenjiawang0312.github.io/">Wenjia Wang</a><sup>1</sup>  
        <a href="https://liangpan99.github.io/">Liang Pan</a><sup>1, 2</sup>  
        <a href="https://frank-zy-dou.github.io/">Zhiyang Dou</a><sup>1</sup>  
        <a href="https://blusque.github.io/">Jidong Mei</a><sup>1</sup>  
        <a href="https://zycliao.com/">Zhouyingcheng Liao</a><sup>1</sup>  
        <br>
        <a href="https://thorin666.github.io/">Yuke Lou</a><sup>1</sup>  
        <a href="https://littlecobber.github.io/">Yifan Wu</a><sup>1</sup>  
        <a href="http://yanglei.me/">Lei Yang</a><sup>2</sup>  
        <a href="https://wangjingbo1219.github.io/">Jingbo Wang</a><sup>2,†</sup>  
        <a href="https://i.cs.hku.hk/~taku/">Taku Komura</a><sup>1,†</sup>
    </p>
    <p>
        <sup>1</sup>The University of Hong Kong    
        <sup>2</sup>Shanghai AI Laboratory
    </p>
    <p>
        <sup>†</sup> Equal Advising
    </p>
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2411.19921" target="_blank">
    <img src="https://img.shields.io/badge/Paper-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="https://wenjiawang0312.github.io/projects/sims/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-F78100?style=plastic&logo=google-chrome&logoColor=white" alt="Project Page">
    </a>
</p>


![teaser](assets/teaser.png)

## 🎬 Overview

SIMS simulates how a character moves through a scene and interacts with furniture in a given style.
With this code, you can train the motion encoder and individual controllers, then run a sequence
such as walking to a chair, sitting down, and standing back up.

## 🚀 Quick Start

1. **Install.** Set up Python 3.8, PyTorch and NVIDIA Isaac Gym using the [installation guide](docs/USAGE.md#installation).
2. **Get the data.** Download the [processed motions](https://huggingface.co/datasets/WenjiaWang/SIMS_data_release) and unpack them in the repository root. See [data preparation](docs/DATA.md) for the scene assets.
3. **Train.** Follow the [training commands](docs/USAGE.md#training) to train the motion encoders first, then the skill controllers.
4. **Run the demo.** Use your trained policies to [make the character walk to a chair, sit down and stand up](docs/USAGE.md#demo).

This release uses hand-written plans. The paper’s automatic script generation is not included.

## 🎓 Citation

If you find this project useful in your research, please consider citing us:

```
@inproceedings{wang2025sims,
title = {SIMS: Simulating Stylized Human-Scene Interactions with Retrieval-Augmented
  Script Generation.},
booktitle = {ICCV},
author = {Wenjia Wang, Liang Pan, Zhiyang Dou, Jidong Mei, Zhouyingcheng Liao, Yuke Lou, Yifan Wu, Lei Yang, Jingbo Wang, Taku Komura.},
year = {2025}
}
```


## 📧 Contact

For questions about SIMS or research collaborations, email Wenjia Wang at wwj2022@connect.hku.hk.
