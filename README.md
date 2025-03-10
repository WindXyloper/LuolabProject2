LuoLab Project 2   2025.3.10 15:41

cresci-2015为加载的数据集(包含edge.csv，id_tweet.json，label.csv，node.json，split.csv，user_info.pt)，processed_data为处理过的数据集（未包含des_tensor.pt和tweet_tensor.pt）。此处二者均省略。

data_process.py为数据处理部分。

model.py为基本模型，包含实验的不同部分，如原始模型、特征ablation和替换为GCN的训练与测试。

self_RGCN_model.py为基本模型修改为自编的RGCN层的模型，仅包含原始模型和特征ablation训练与测试。

基于整个实验是在远程hpc上运行，所有的训练结果存储在训练日志内。

当前版本tweet处理还未完成，完成后的tweet处理版本会于后续上传。
