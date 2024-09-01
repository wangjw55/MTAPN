# Multi-Task Multi-Agent Shared Layers are Universal Cognition of Multi-Agent Coordination
The master branch is utilized for single-task training, the football branch for multi-task pre-training in the GRF environment, and the multienv branch for multi-task pre-training in the SMAC environment.

## Video
Video a: https://youtu.be/Ng1axoM43j8

Video b: https://youtu.be/HCamoGRcRs0

Video c: https://youtu.be/HF8Bj9lVdlI

Video d: https://youtu.be/M-0BlV-cSxE

Video e: https://youtu.be/E2PG1Mejf3c

Video f: https://youtu.be/MGV8PA71x8s

Video a shows the training performance of 27m_vs_30m in multi-task pre-training. Video b demonstrates the performance of 8m_vs_9m when trained up to 6M steps using pre-trained DecL. Video c presents the performance of 8m_vs_9m when trained from scratch up to 6M steps. In video a, all Marines position themselves effectively during the attack, focusing their firepower on a single enemy, which rapidly reduces the number of enemies. In video b, the Marines similarly adopt a concentrated fire strategy, successfully gaining a numerical advantage. However, in video c, the Marines attack two enemy Marines simultaneously. Due to their numerical disadvantage and dispersed firepower, they are ultimately defeated.

Video d shows the training performance of 5m_vs_6m in multi-task pre-training. Video e demonstrates the performance of MMM2 when trained up to 10M steps using pre-trained DecL. Video f presents the performance of MMM2 when trained from scratch up to 10M steps. In video d, a Marine retreats promptly when its health is low, successfully diverting enemy fire to other teammates with higher health. In video e, the Medivac also employs a retreat strategy, ensuring the continuous recovery of the health of other agents. However, in video f, the Medivac fails to adopt an effective evasion strategy, leading to its loss and preventing the timely recovery of the health of other agents, ultimately resulting in defeat.

## Citation
If you find this work useful, please consider citing:
```
@article{wang2023multi,
  title={Multi-Task Multi-Agent Shared Layers are Universal Cognition of Multi-Agent Coordination},
  author={Wang, Jiawei and Zhao, Jian and Cao, Zhengtao and Feng, Ruili and Qin, Rongjun and Yu, Yang},
  journal={arXiv preprint arXiv:2312.15674},
  year={2023}
}
```
