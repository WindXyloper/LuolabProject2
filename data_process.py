# data_process.py
import os
import json
import pandas as pd
import torch
from datetime import datetime
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaModel

def process_data(raw_dir="cresci-2015", processed_dir="processed_data"):
    # 创建输出目录
    os.makedirs(processed_dir, exist_ok=True)
    
    # ================== 处理节点映射 ==================
    print("Processing node mapping...")
    with open(f"{raw_dir}/node.json", "r") as f:
        nodes = json.load(f)  # 直接加载整个数组
    id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)

    # # ================== 处理边数据 ==================
    print("Processing edges...")
    edges_df = pd.read_csv(f"{raw_dir}/edge.csv")
    valid_edges = edges_df[edges_df['relation'].isin(['follow', 'friend'])]

    edge_index, edge_type = [], []
    for _, row in valid_edges.iterrows():
        src, rel, tgt = row['source_id'], row['relation'], row['target_id']
        if src not in id_to_idx or tgt not in id_to_idx:
            continue
        src_idx, tgt_idx = id_to_idx[src], id_to_idx[tgt]
        
        if rel == 'follow':
            edge_index.append([src_idx, tgt_idx])
            edge_type.append(0)
        elif rel == 'friend':
            edge_index.extend([[src_idx, tgt_idx], [tgt_idx, src_idx]])
            edge_type.extend([1, 1])

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
    torch.save(edge_index_tensor, f"{processed_dir}/edge_index.pt")
    torch.save(edge_type_tensor, f"{processed_dir}/edge_type.pt")
    
    # ================== 处理数据集划分 ==================
    print("Processing splits...")
    split_df = pd.read_csv(f"{raw_dir}/split.csv")
    split_df['index'] = split_df['id'].map(id_to_idx).dropna().astype(int)

    for split in ['train', 'val', 'test']:
        indices = split_df[split_df['split'] == split]['index'].tolist()
        torch.save(torch.tensor(indices, dtype=torch.long), 
                  f"{processed_dir}/{split}_index.pt")

    # ================== 处理标签 ==================
    print("Processing labels...")
    label_df = pd.read_csv(f"{raw_dir}/label.csv")
    label_df['index'] = label_df['id'].map(id_to_idx).dropna()
    label_df = label_df.sort_values('index').dropna()
    
    # 保存为CSV和Tensor双格式
    label_df[['id', 'label']].to_csv(f"{processed_dir}/label.csv", index=False)
    labels = label_df['label'].map({'human': 0, 'bot': 1}).values
    torch.save(torch.tensor(labels, dtype=torch.long), f"{processed_dir}/label.pt")

    print(f"Processing complete! Output saved to {processed_dir}")

    # ================== 处理特征 ==================
    print("Processing features...")
    # 初始化RoBERTa模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta_path')   #可能会有错误。
    model = RobertaModel.from_pretrained('roberta_path').to(device)
    model.eval()

    # ------------------ 处理描述特征 ------------------
    print("Processing description features...")

    def encode_description(text):
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            max_length=128,          # 限制最大长度
            truncation=True,
            padding="max_length"     # 填充到固定长度
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

    # 分批处理（关键修改）
    batch_size = 32768  # 根据可用内存调整
    des_tensors = []

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        batch_des = [str(node.get('description', '')) for node in batch]
        
        # 编码并立即释放资源
        batch_tensor = torch.stack([encode_description(desc) for desc in batch_des])
        des_tensors.append(batch_tensor)
        
        # 显式释放内存
        del batch, batch_des, batch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Processed {min(i+batch_size, len(nodes))}/{len(nodes)}")

    # 合并并保存
    des_tensor = torch.cat(des_tensors, dim=0)
    torch.save(des_tensor, f"{processed_dir}/des_tensor.pt")

    # ------------------ 处理推文特征 ------------------
    print("Processing tweet features...")

    with open(f"{raw_dir}/id_tweet.json", "r") as f:
        tweets_data = json.load(f)

    tweet_embeddings = []
    for idx in range(num_nodes):
        user_tweets = tweets_data.get(str(idx), [])
        
        if len(user_tweets) == 0:
            avg_embedding = torch.zeros(768)
        else:
            embeddings = []
            for tweet in user_tweets:
                if not str(tweet).strip():
                    embeddings.append(torch.zeros(768))
                    continue
                
                inputs = tokenizer(str(tweet), return_tensors="pt",
                                max_length=128, truncation=True,
                                padding='max_length').to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu())
            
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0) if embeddings else torch.zeros(768)
        
        tweet_embeddings.append(avg_embedding)

    tweets_tensor = torch.stack(tweet_embeddings)
    torch.save(tweets_tensor, f"{processed_dir}/tweets_tensor.pt")

    # ------------------ 处理数值特征 ------------------
    print("Processing numerical features...")

    from datetime import datetime

    num_features = []
    for node in nodes:
        metrics = node.get('public_metrics', {})
        
        # 计算活跃天数
        try:
            created_at = datetime.strptime(node['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
            active_days = (datetime.now() - created_at).days
        except:
            active_days = 0  # 异常处理
        
        num_feat = [
            metrics.get('followers_count', 0),      # 粉丝数
            metrics.get('following_count', 0),      # 关注数
            metrics.get('tweet_count', 0),          # 推文数（对应论文的statuses）
            active_days,                            # 活跃天数
            len(str(node.get('username', '')))      # 用户名长度
        ]
        num_features.append(num_feat)

    num_tensor = torch.tensor(num_features, dtype=torch.float)
    num_mean = num_tensor.mean(dim=0)
    num_std = num_tensor.std(dim=0)
    num_std[num_std == 0] = 1e-6  # 防止除零
    num_normalized = (num_tensor - num_mean) / num_std
    torch.save(num_normalized, f"{processed_dir}/num_tensor.pt")

    # ------------------ 处理分类特征 ------------------
    print("Processing categorical features...")

    cat_features = []
    for node in nodes:
        # 原始特征提取
        protected = node.get('protected', False) or False
        verified = node.get('verified', False) or False
        
        # 用location存在性模拟geo_enabled
        has_location = bool(node.get('location')) and (str(node['location']).strip() != "")
        
        # One-hot编码（每个特征扩展为2维）
        protected_onehot = [1, 0] if protected else [0, 1]
        verified_onehot = [1, 0] if verified else [0, 1]
        geo_onehot = [1, 0] if has_location else [0, 1]
        
        cat_features.append(protected_onehot + verified_onehot + geo_onehot)

    cat_tensor = torch.tensor(cat_features, dtype=torch.float)
    torch.save(cat_tensor, f"{processed_dir}/cat_tensor.pt")
    

if __name__ == "__main__":
    process_data()


'''
node.json例子：
{"created_at": "Fri Apr 06 10:58:22 +0000 2007", "description": "Founder of http://www.screenweek.it & http://www.boxofficecup.com - Apple & Movie Lover find me on rebelmouse http://www.rebelmouse.com/davidedellacasa", "entities": null, "id": "u3610511", "location": "Roma", "name": "Davide Dellacasa", "pinned_tweet_id": null, "profile_image_url": "http://a0.twimg.com/profile_images/1575057050/Stay_hungry._Stay_foolish_Avatar_normal.png", "protected": null, "public_metrics": {"followers_count": 5470, "following_count": 2385, "tweet_count": 20370, "listed_count": 52}, "url": "http://braddd.tumblr.com", "username": "braddd", "verified": null, "withheld": null}
'''