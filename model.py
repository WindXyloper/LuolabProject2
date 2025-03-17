import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from sklearn.metrics import f1_score

class BotRGCN(nn.Module):
    def __init__(self, 
                 des_dim=768, 
                 tweet_dim=768, 
                 num_dim=5, 
                 cat_dim=6,
                 hidden_dim=128,
                 num_relations=2,
                 num_layers=2,
                 dropout=0.3,
                 use_des=True, 
                 use_tweet=True,
                 use_num=True,
                 use_cat=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
                     
        self.use_des = use_des
        self.use_tweet = use_tweet
        self.use_num = use_num
        self.use_cat = use_cat
        
        self.des_proj = nn.Linear(des_dim, hidden_dim)
        self.tweet_proj = nn.Linear(tweet_dim, hidden_dim)
        self.num_proj = nn.Linear(num_dim, hidden_dim)
        self.cat_proj = nn.Linear(cat_dim, hidden_dim)
        
        self.rgcns = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=None)
            for _ in range(num_layers)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.dropout = dropout

    def forward(self, data):    #！！！！
        x = 0
        if self.use_des:
            h_des = F.leaky_relu(self.des_proj(data.des))
            x += h_des
        if self.use_tweet:
            h_tweet = F.leaky_relu(self.tweet_proj(data.tweets))
            x += h_tweet
        if self.use_num:
            h_num = F.leaky_relu(self.num_proj(data.num))
            x += h_num
        if self.use_cat:
            h_cat = F.leaky_relu(self.cat_proj(data.cat))
            x += h_cat
        
        for rgcn in self.rgcns:
            x = F.leaky_relu(rgcn(x, data.edge_index, data.edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.mlp(x)

class BotDetector:
    def __init__(self, processed_dir='processed_data'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.load_data(processed_dir).to(self.device)
        
    def load_data(self, dir_path):
        return Data(
            des=torch.load(f"{dir_path}/des_tensor.pt"),
            tweets=torch.load(f"{dir_path}/tweets_tensor.pt"),
            num=torch.load(f"{dir_path}/num_tensor.pt"),
            cat=torch.load(f"{dir_path}/cat_tensor.pt"),
            edge_index=torch.load(f"{dir_path}/edge_index.pt"),
            edge_type=torch.load(f"{dir_path}/edge_type.pt"),
            y=torch.load(f"{dir_path}/label.pt"),
            train_idx=torch.load(f"{dir_path}/train_index.pt"),
            val_idx=torch.load(f"{dir_path}/val_index.pt"),
            test_idx=torch.load(f"{dir_path}/test_index.pt")
        )
    
    def train(self, model, epochs=200, lr=0.01, patience=20):
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=lr,
                                   weight_decay=1e-4)
        best_val_acc = 0
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(self.data)
            loss = F.cross_entropy(out[self.data.train_idx], 
                                 self.data.y[self.data.train_idx])
            
            loss.backward()
            optimizer.step()
            
            # 验证
            val_acc, val_f1 = self.evaluate(model, 'val')
            print(f'Epoch {epoch+1:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
            
            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                no_improve += 1
                if no_improve == patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_model.pth'))
        test_acc, test_f1 = self.evaluate(model, 'test') 
        print(f'Final Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}') 
    
    def evaluate(self, model, split='test'):
        model.eval()
        with torch.no_grad():
            out = model(self.data)
            pred = out.argmax(dim=1)
            mask = getattr(self.data, f'{split}_idx')
            y_true = self.data.y[mask].cpu()
            y_pred = pred[mask].cpu()
            acc = (y_pred == y_true).sum().item() / mask.size(0)
            f1 = f1_score(y_true, y_pred, average='macro')
            return acc, f1 

def main():
    detector = BotDetector()
    
    print("Training Original BotRGCN:")
    model = BotRGCN(num_layers=2).to(detector.device)
    detector.train(model)
    
    # 消融实验示例：不同层数
    print("\nAblation Study - 1 Layer:")
    model_1layer = BotRGCN(num_layers=1).to(detector.device)
    detector.train(model_1layer)
    
    # 消融实验 - 特征集
    print("\nAblation - Text Features Only:")
    model_text = BotRGCN(use_num=False, use_cat=False).to(detector.device)
    detector.train(model_text)
    
    # 替换为GCN
    from torch_geometric.nn import GCNConv
    
    class BotGCN(BotRGCN):
        """GCN版本实现"""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # 覆盖RGCN层为GCN层
            self.rgcns = nn.ModuleList([
                GCNConv(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layers)
            ])
        
        def forward(self, data):
            # 特征融合（复用父类的特征投影）
            x = 0
            if self.use_des:
                h_des = F.leaky_relu(self.des_proj(data.des))
                x += h_des
            if self.use_tweet:
                h_tweet = F.leaky_relu(self.tweet_proj(data.tweets))
                x += h_tweet
            if self.use_num:
                h_num = F.leaky_relu(self.num_proj(data.num))
                x += h_num
            if self.use_cat:
                h_cat = F.leaky_relu(self.cat_proj(data.cat))
                x += h_cat
            
            # 多层GCN处理
            for gcn in self.rgcns:
                x = F.leaky_relu(gcn(x, data.edge_index))  # 仅传递edge_index
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            return self.mlp(x)

    print("Training BotGCN:")
    model = BotGCN(  # 显式设置参数
        hidden_dim=64,
        num_layers=1,
        use_des=True,
        use_num=True,
        use_cat=True
    ).to(detector.device)
    detector.train(model)

if __name__ == "__main__":
    main()
