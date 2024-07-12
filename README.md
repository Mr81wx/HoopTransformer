# HoopTransformer
Paper link https://link.springer.com/article/10.1007/s40279-024-02030-3

Understanding and recognizing basketball offensive plays, which involve intricate interactions between players, have always been regarded as challenging tasks for untrained humans, not to mention machines. Inspired by the Large language Model like ChatGPT and BERT, We aim to train a pre-training model to automatically recognize offensive plays by proposing a novel self-supervised learning model for trajectory prediction. 

Model Architecture:
The model is based on the encoder-decoder architecture and employ combination of motion prediction and motion reconstruction for predicting players' high-velocity trajectory. Pre-training on more than 90,000+ offensive possessions.

#Data Sportvu 2015-16 
  - invalid data (missing 24s): 0021500040; 0021500046; 0021500050; 0021500052; 002150065; 0021500109; 0021500129; 0021500237; 0021500543; 0021500599; 0021500646; 0021500652;

  - 
Results:
https://sites.google.com/view/hoop-transformer/%E9%A6%96%E9%A1%B5

Cite this article
Wang, X., Tang, Z., Shao, J. et al. HoopTransformer: Advancing NBA Offensive Play Recognition with Self-Supervised Learning from Player Trajectories. Sports Med (2024). https://doi.org/10.1007/s40279-024-02030-3
