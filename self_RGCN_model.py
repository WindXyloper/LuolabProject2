import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.metrics import f1_score

class SimpleRGCNConv(MessagePassing):
    """内存安全的RGCN实现"""
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__(aggr='mean')
        self.num_relations = num_relations
        
        # 更紧凑的参数初始化
        self.weight = nn.Parameter(
            torch.Tensor(num_relations, in_channels, out_channels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x, edge_index, edge_type):
        # 分关系类型处理避免内存爆炸
        out = torch.zeros(x.size(0), self.weight.size(0), self.weight.size(2), 
                         device=x.device)
        
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() > 0:
                x_r = F.linear(x, self.weight[r])
                out[:, r] = self.propagate(
                    edge_index[:, mask],
                    x=x_r,
                    size=(x.size(0), x.size(0))
                )
        return out.mean(dim=1)  # 聚合所有关系

    def message(self, x_j):
        return x_j

class BotRGCN(nn.Module):
    def __init__(self, 
                 des_dim=768,
                 num_dim=5, 
                 cat_dim=6,
                 hidden_dim=64,  # 降低维度
                 num_relations=2,
                 num_layers=1,   # 减少层数
                 dropout=0.5,    # 增加dropout
                 **kwargs):
        super().__init__()
        
        # 带归一化的投影层
        self.des_proj = nn.Sequential(
            nn.Linear(des_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 单层RGCN
        self.rgcn = SimpleRGCNConv(hidden_dim, hidden_dim, num_relations)
        
        # 更紧凑的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data):
        # 特征融合
        x = 0
        x += F.leaky_relu(self.des_proj(data.des))
        x += F.leaky_relu(self.num_proj(data.num))
        x += F.leaky_relu(self.cat_proj(data.cat))
        
        # RGCN处理
        x = F.leaky_relu(self.rgcn(x, data.edge_index, data.edge_type))
        x = F.dropout(x, p=0.5, training=self.training)
        
        return self.classifier(x)

class BotDetector:
    def __init__(self, processed_dir='processed_data'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self.load_data(processed_dir)
        self.validate_data()
        self.data = self.data.to(self.device)

    def validate_data(self):
        """增强数据验证"""
        print("\n=== Enhanced Data Validation ===")
        assert self.data.edge_index.dim() == 2, "Edge index must be 2D tensor"
        assert self.data.edge_type.max() < 2, "Edge type should be 0 or 1"
        
        # 显式设置节点数
        self.data.num_nodes = self.data.des.size(0)
        print(f"Confirmed nodes: {self.data.num_nodes}")
        print(f"Edge index range: [{self.data.edge_index.min()}, {self.data.edge_index.max()}]")
        
    def load_data(self, dir_path):
        """显式设置所有必要属性"""
        data_dict = {
            'des': torch.load(f"{dir_path}/des_tensor.pt"),
            'num': torch.load(f"{dir_path}/num_tensor.pt"),
            'cat': torch.load(f"{dir_path}/cat_tensor.pt"),
            'edge_index': torch.load(f"{dir_path}/edge_index.pt"),
            'edge_type': torch.load(f"{dir_path}/edge_type.pt"),
            'y': torch.load(f"{dir_path}/label.pt"),
            'train_idx': torch.load(f"{dir_path}/train_index.pt"),
            'val_idx': torch.load(f"{dir_path}/val_index.pt"),
            'test_idx': torch.load(f"{dir_path}/test_index.pt"),
            'num_nodes': torch.load(f"{dir_path}/des_tensor.pt").size(0)  # 关键修复
        }
        return Data(**data_dict)
    
    def train(self, model, epochs=100, lr=0.005, patience=10):
        """优化训练过程"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        scaler = torch.cuda.amp.GradScaler()  # 混合精度
        
        best_val_acc = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # 混合精度训练
            with torch.cuda.amp.autocast():
                out = model(self.data)
                loss = F.cross_entropy(out[self.data.train_idx], self.data.y[self.data.train_idx])
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            
            # 学习率调整
            val_acc, val_f1 = self.evaluate(model, 'val')
            scheduler.step(val_acc)
            
            print(f'Epoch {epoch+1:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
            
            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                
            if epoch - best_val_acc >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        model.load_state_dict(torch.load('best_model.pth'))
        test_acc, test_f1 = self.evaluate(model, 'test')
        print(f'Final Test | Acc: {test_acc:.4f} | F1: {test_f1:.4f}')

    def evaluate(self, model, split='test'):
        model.eval()
        with torch.no_grad():
            out = model(self.data)
            pred = out.argmax(dim=1)
            mask = getattr(self.data, f'{split}_idx')
            return self._calc_metrics(pred[mask], self.data.y[mask])

    def _calc_metrics(self, pred, target):
        acc = (pred == target).float().mean().item()
        f1 = f1_score(target.cpu(), pred.cpu(), average='macro')
        return acc, f1

def main():
    detector = BotDetector()
    
    print("Training Optimized Model:")
    model = BotRGCN(hidden_dim=64, num_layers=1).to(detector.device)
    
    # 内存监控
    print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    detector.train(model, epochs=50, lr=0.001)

if __name__ == "__main__":
    main()
