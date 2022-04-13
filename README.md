# Deep Learning-Based Power Control for Uplink Cell-Free Massive MIMO Systems

This is a code package related to the following scientific article:

Y. Zhang, J. Zhang, Y. Jin, S. Buzzi, and B. Ai, “Deep learning-based power control for uplink cell-free massive MIMO systems,” in *2021 IEEE Globecom.* IEEE, 2021, pp. 1–6.

The package contains a simulation environment, based on Matlab and Python, that reproduces some of the numerical results and figures in the article. We encourage you to also perform reproducible research!

## Abstract of Article

In this paper, a general framework for deep learning-based power control methods for max-min, max-product and max-sum-rate optimization in uplink cell-free massive multiple-input multiple-output (CF mMIMO) systems is proposed. Instead of using supervised learning, the proposed method relies on unsupervised learning, in which optimal power allocations are not required to be known, and thus has low training complexity. More specifically, a deep neural network (DNN) is trained to learn the map between fading coefficients and power coefficients within short time and with low computational complexity. It is interesting to note that the spectral efficiency of CF mMIMO systems with the proposed method outperforms previous optimization methods for max-min optimization and fits well for both max-sum-rate and max-product optimizations.

## Content of Code Package

The package generates the simulation results:

- `DL_Maxmin`: Neural network for maxmin optimization
- `DL_Maxprod`: Neural network for maxprod optimization
- `DL_Maxsum`: Neural network for maxsum optimization

Other python and matlab codes are also included in the folder. See each file for further documentation.

## License and Referencing

This code package is licensed under the GPLv2 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

