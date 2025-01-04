# LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation

## 💻 Installation Guide

1. **Create the environment**:

   ```bash
   make up -d train
   conda create -n llm python=3.8
   conda activate llm #source /opt/conda/bin/activate llm
   pip install -r requirements.txt

   ## Install apex
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

   ## Recommended Version
   torch==2.3.0

   ## Install latest llm2vec library
   git clone https://github.com/McGill-NLP/llm2vec.git
   cd llm2vec
   pip install -e .
   ```
2. **Data Preparation**:

   *(Coming Soon)*

3. **🔥 Training**:

   ```bash
   sh run.sh
   ```

## ❤️ Acknowledgements

Our code is built on top of [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP). We would like to thank the EVA team for their foundational work.

## Citation

If you use our work, please cite:

```
@misc{huang2024llm2clippowerfullanguagemodel,
      title={LLM2CLIP: Powerful Language Model Unlock Richer Visual Representation}, 
      author={Weiquan Huang and Aoqi Wu and Yifan Yang and Xufang Luo and Yuqing Yang and Liang Hu and Qi Dai and Xiyang Dai and Dongdong Chen and Chong Luo and Lili Qiu},
      year={2024},
      eprint={2411.04997},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.04997}, 
}
