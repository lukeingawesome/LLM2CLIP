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
   ```
2. **Data Preparation**:

   *(Coming Soon)*

3. **🔥 Training**:

   ```bash
   sh run.sh
   ```

# 📚 FAQ
For more insights and answers, visit our [FAQ](FAQ.md).
## Q1:

> **Q: It is foreseeable that the technology of LLM2CLIP will be of great significance in expanding CLIP's support for more modal data. As far as the article is concerned, LLM2CLIP has surprisingly improved CLIP's adaptability to cross-language and long text tasks. At the same time, it also proposes application possibilities for higher-dimensional data modalities such as audio and video. Of course, this puts forward further requirements for LLM2CLIP's adaptation strategy and fine-tuning methods. Based on your team's current understanding of LLM2CLIP, what additional challenges will arise, for example, the feature space alignment problem of high-dimensional modalities?**

> ![A1](https://via.placeholder.com/15/blue/000000?text=+) **A:** To be honest, we’re already exploring a video-based version of LLM2CLIP, including scaling up both the dataset size and model parameters by several orders of magnitude. Please stay tuned for our future updates, and if you’re interested, we’d be happy to discuss this further!
>
> Here are some additional challenges I see in this area:
>
> 1. **Enhancing the Supervisory Signal in Contrastive Learning:** While LLMs have a strong capability to understand text, providing valuable and rich textual information is equally critical. For instance, for video tasks, we could enrich the input with denser captions, prompts, or instructions. These could provide more complex and detailed information for the LLM to interpret, thereby enabling it to better guide the construction of the cross-modal space.
> 
> 2. **Expanding Contrastive Learning Loss Across Dimensions:** Contrastive learning losses can be applied across various dimensions, such as the temporal dimension in video data. Different prompts provided to the LLM could be designed to guide and control the training process in these additional dimensions, further strengthening the multimodal representations.
>
> 3. **Tackling Complex Temporal Logic in Videos:** The challenges in video understanding often involve designing solutions for complex temporal relationships over extended time spans. Here, we could incorporate self-play techniques using the LLM to introduce tasks and increase the complexity of the training objectives. This might involve designing scenarios where the LLM can simulate and reason about sequences, further enhancing its learning.

## Q2:

> **Q: What a groundbreaking paper on LLM2CLIP! The innovative integration of large language models with CLIP to enhance cross-modal representation learning is truly inspiring. The performance improvements demonstrated, particularly in long-text and short-text retrieval tasks, are impressive and have significant implications for the field of multimodal AI.**
>
> **My admiration for your work encourages me to inquire about the potential applications of LLM2CLIP in more specialized domains, such as medicine or law, where the precision and expertise of textual understanding are paramount. Therefore, I am curious to know if LLM2CLIP has been tested or if there are plans to test it with domain-specific texts that require a high degree of accuracy and proficiency.**
>
> Looking forward to your insights on this matter and how LLM2CLIP might be adapted or extended to meet the challenges of these specialized fields!
>
> ![A2](https://via.placeholder.com/15/green/000000?text=+) **A:** Your idea is fantastic, and in fact, we have had similar thoughts. I believe there is significant potential in working on specialized fields, and here are my reasons:
>
> 1. **Limited Data, High Impact:** Our work focuses on fine-tuning pre-trained CLIP models with very limited data for LLM2CLIP, ranging from 3M to 60M. Compared to the 1-2B data commonly used in CLIP pre-training, this is a small amount, yet it has already demonstrated substantial performance improvements. If we focus on specialized fields, we could leverage limited domain-specific data to train the model exceptionally well in a specific knowledge area. This approach could potentially resolve issues like perception or cognition hallucinations in related multimodal domains entirely.
>
> 2. **Leveraging LLM Knowledge as Data Augmentation:** Certain specialized fields, such as medical reports, often suffer from a lack of data. Here, the knowledge encoded in LLMs can serve as an excellent data augmenter due to their access to open-world knowledge over time.
>
> We look forward to collaborating with you to push the boundaries of multimodal domains!
>
> BTW, we plan to release scaled-up LLM2CLIP models (10-100x larger) next quarter. These models will inherit our general-purpose parameters, potentially making them even more powerful. Please stay tuned to our GitHub!

## Q3:

> **Q: Thank you so much for such an outstanding work. I have a couple of questions regarding the fine-tuning process described in Section 3.2, particularly around the integration of loss functions and datasets:**
>
> **In the paper, two loss functions are mentioned: SimCSE loss and Masked Next Token Prediction (MNTP). However, it is unclear whether these two loss functions are used simultaneously during training, or if the training process is split into different phases where each loss is applied separately. Could you please clarify how the losses are used? If they are used together, what are the relative weights assigned to each?**
>
> **Regarding the datasets, CC-3M and Wikitext-103 are mentioned as part of the training process. It seems a bit unclear how these two datasets are combined in the training phase. Given that Wikitext-103 is a pure language corpus while CC-3M is image-caption based, how are they jointly used during the fine-tuning process? Are they used for different stages or tasks?**
>
> Looking forward to your insights on this!
>
> ![A3](https://via.placeholder.com/15/red/000000?text=+) **A:** Thank you for your question. I’m glad to clarify.
>
> **Loss Functions Integration:** We use the supervised SimCSE loss to make different captions of the same image positive samples for each other, while captions of different images serve as negative samples. This loss function is key to our method, allowing the LLM to provide meaningful supervisory signals to the image. However, the Masked Next Token Prediction (MNTP) was an initial stage we employed before using the supervised SimCSE loss; it can be understood as an earlier step in training. We first conduct MNTP, followed by supervised SimCSE loss, in a two-stage process. In practice, MNTP has little impact on the results, so removing it does not affect the conclusions. However, for optimal performance, we still chose to use MNTP before applying supervised SimCSE loss.
>
> **Dataset Combination:** We indeed mix both pure text and caption datasets. This is because the LLM is initially pre-trained on pure text data, so we aim to retain its original distribution with minimal shift by using the pure text dataset Wikitext-103, which also helps mitigate any bias introduced by captions. Our approach is to mix and shuffle the two datasets and then sample batches normally for training. This is a common and effective practice.
>
> If you have more questions, please feel free to ask.
> 
> 
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
