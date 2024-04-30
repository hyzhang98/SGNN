# Decouple Graph Neural Networks: Train Multiple Simple GNNs Simultaneously Instead of One

This repository is our implementation of 

>   Hongyuan Zhang, Yanan Zhu, and Xuelong Li,  "Decouple Graph Neural Networks: Train Multiple Simple GNNs Simultaneously Instead of One," *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, DOI: 10.1109/TPAMI.2024.3392782, 2024.[(arXiv)](https://arxiv.org/pdf/2304.10126.pdf)[(IEEE)](https://ieeexplore.ieee.org/document/10507024)

*SGNN* attempts to further reduce the training complexity of each iteration from $\mathcal{O}(n^2) / \mathcal{O}(|\mathcal E|)$ (vanilla GNNs without acceleration tricks, e.g., [AdaGAE](https://github.com/hyzhang98/AdaGAE)) and $\mathcal O(n)$ (e.g., [AnchorGAE](https://github.com/hyzhang98/AnchorGAE-torch)) to $\mathcal O(m)$. 

Compared with other fast GNNs, SGNN can

-   (**Exact**) compute representations exactly (without sampling);
-   (**Non-linear**) use up to $L$ non-linear activations ($L$ is the number of layers);
-   (**Fast**) be trained with the real stochastic (mini-batch based) optimization algorithms. 

The comparison is summarized in the following table. 


![Comparison](figures/Comparison.jpg)



If you have issues, please email:

hyzhang98@gmail.com



## Requirements 

- pytorch 1.10.0
- scipy 1.3.1
- scikit-learn 0.21.3
- numpy 1.16.5



## How to run SGNN

>   Please ensure the data is rightly loaded

```
python run.py
python run_classfication.py
```



## Citation
```
@article{SGNN,
  author={Zhang, Hongyuan and Zhu, Yanan and Li, Xuelong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Decouple Graph Neural Networks: Train Multiple Simple GNNs Simultaneously Instead of One}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2024.3392782}
}
```