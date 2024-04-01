# Multi-Task Multi-Agent Shared Layers are Universal Cognition of Multi-Agent Coordination
The master branch is utilized for single-task training, the football branch for multi-task pre-training in the GRF environment, and the multienv branch for multi-task pre-training in the SMAC environment.

## Video
Video a: https://youtu.be/FHK-W4hH0DA

Video b: https://youtu.be/MGV8PA71x8s 

Video c: https://youtu.be/E2PG1Mejf3c

Video a shows the training performance of 3s_vs_5z in multi-task pre-training. Video b presents the performance of MMM2 when trained from scratch up to 10M steps. Video c demonstrates the performance of MMM2 when trained up to 10M steps using pre-trained DecL.

In video a, we observe that our method learns effective policies for 3s_vs_5z during multi-task pre-training. After each Stalker's attack on Zealots, they quickly retreat to avoid Zealots' counterattacks. Additionally, some Stalkers draw enemy attention while others cooperate to defeat a small group of Zealots.

In video b, the Medivac fails to adopt an evasion strategy to avoid enemy attacks, resulting in the loss of our Medivac and subsequently leaving other agents on our side unable to recover their health in time. In addition, the positioning of Marines and Marauders lacks strategy as they fail to retreat in timewhen their health is low.

In video c, we observe that the agent learns some advanced policies. Marines with low health retreat in time, while those with higher health draw fire from enemy Marauders. Meanwhile, other Marines stealthily flank the enemy Marauders for surprise attacks.

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
