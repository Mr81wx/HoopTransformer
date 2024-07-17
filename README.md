# HoopTransformer
Paper link https://link.springer.com/article/10.1007/s40279-024-02030-3

Understanding and recognizing basketball offensive plays, which involve intricate interactions between players, have always been regarded as challenging tasks for untrained humans, not to mention machines. Inspired by the Large language Model like ChatGPT and BERT, We aim to train a pre-training model to automatically recognize offensive plays by proposing a novel self-supervised learning model for trajectory prediction. 

Application:
Based on the Encoder, we can genarate Possession Embedding for each possession (vector database), and use similarity search to find the possessions run same set play.
https://github.com/user-attachments/assets/0e82a76c-60ca-4281-8b73-5aba5cd2298d

Model Architecture:
The model is based on the encoder-decoder architecture and employ combination of motion prediction and motion reconstruction for predicting players' high-velocity trajectory. Pre-training on more than 90,000+ offensive possessions.
![F1_00](https://github.com/user-attachments/assets/e36aa4ac-b322-4bcd-b575-6333da49fe4f)

Masking strategy：
![F2_00](https://github.com/user-attachments/assets/9fa83a8a-e858-4f11-9631-7d89d9a99ae1)

Results:
![F3_00](https://github.com/user-attachments/assets/e320edb2-849f-4157-9512-d63f5c8166df)




#Data Sportvu 2015-16 
  - invalid data (missing 24s): 0021500040; 0021500046; 0021500050; 0021500052; 002150065; 0021500109; 0021500129; 0021500237; 0021500543; 0021500599; 0021500646; 0021500652;

    
Cite this article
Wang, X., Tang, Z., Shao, J. et al. HoopTransformer: Advancing NBA Offensive Play Recognition with Self-Supervised Learning from Player Trajectories. Sports Med (2024). https://doi.org/10.1007/s40279-024-02030-3







