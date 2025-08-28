import torch
from torch.utils.data import Dataset
from .gen_dyck import DyckLanguage, RDyck1Language, generate_dyck_dataset, generate_rdyck_dataset


class DyckDataset(Dataset):
    """
    Dyck语言数据集类
    与FormalLanguageDataset保持一致的接口
    """
    
    def __init__(self, data, vocab=None):
        """
        初始化Dyck数据集
        
        Args:
            data: 数据列表，每个元素为(输入字符串, 输出标签)
            vocab: 词汇表，如果为None则自动创建
        """
        self.data = data
        
        # 如果没有提供词汇表，则从数据中创建
        if vocab is None:
            self.vocab = self._create_vocab()
        else:
            self.vocab = vocab
    
    def _create_vocab(self):
        """
        从数据中创建词汇表
        
        Returns:
            词汇表字典
        """
        vocab = {}
        char_set = set()
        
        # 收集所有唯一的字符
        for x_str, _ in self.data:
            char_set.update(x_str)
        
        # 创建词汇表映射
        for i, char in enumerate(sorted(char_set)):
            vocab[char] = i
        
        return vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x_str, y = self.data[idx]
        
        # 将输入字符串转换为张量
        x_tensor = torch.tensor([self.vocab[c] for c in x_str], dtype=torch.long)
        
        # 将输出标签转换为张量
        if isinstance(y, str):
            y_tensor = torch.tensor([self.vocab[c] for c in y], dtype=torch.long)
        else:
            # y是one-hot编码的列表
            y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor


class DyckLanguageDataset(DyckDataset):
    """
    标准Dyck语言数据集
    """
    
    def __init__(self, n_samples=10000, seq_len=16, num_pairs=2, p=0.4, q=0.3, vocab=None):
        """
        初始化Dyck语言数据集
        
        Args:
            n_samples: 样本数量
            seq_len: 序列长度
            num_pairs: 括号对数量
            p: 嵌套概率
            q: 连接概率
            vocab: 词汇表
        """
        data = generate_dyck_dataset(n_samples, seq_len, num_pairs, p, q)
        super().__init__(data, vocab)


class RDyckDataset(DyckDataset):
    """
    重置Dyck语言数据集
    """
    
    def __init__(self, n_samples=10000, seq_len=16, p=0.4, q=0.3, vocab=None):
        """
        初始化重置Dyck语言数据集
        
        Args:
            n_samples: 样本数量
            seq_len: 序列长度
            p: 嵌套概率
            q: 连接概率
            vocab: 词汇表
        """
        data = generate_rdyck_dataset(n_samples, seq_len, p, q)
        super().__init__(data, vocab)


def create_dyck_vocab(num_pairs=2):
    """
    创建Dyck语言的词汇表
    
    Args:
        num_pairs: 括号对数量
        
    Returns:
        词汇表字典
    """
    all_pairs = ['()', '[]', '{}', '<>', '+-', 'ab', 'xo']
    pairs = all_pairs[:num_pairs]
    
    vocab = {}
    for i, pair in enumerate(pairs):
        vocab[pair[0]] = i * 2      # 开括号
        vocab[pair[1]] = i * 2 + 1  # 闭括号
    
    return vocab


def create_rdyck_vocab():
    """
    创建重置Dyck语言的词汇表
    
    Returns:
        词汇表字典
    """
    vocab = {'(': 0, ')': 1, '1': 2}  # 重置符号
    return vocab


def get_dyck_dataset(n_samples=3000, seq_len=16, num_pairs=2, p=0.4, q=0.3):
    """
    获取Dyck语言数据集的便捷函数
    
    Args:
        n_samples: 样本数量
        seq_len: 序列长度
        num_pairs: 括号对数量
        p: 嵌套概率
        q: 连接概率
        
    Returns:
        DyckDataset实例
    """
    vocab = create_dyck_vocab(num_pairs)
    return DyckLanguageDataset(n_samples, seq_len, num_pairs, p, q, vocab)


def get_rdyck_dataset(n_samples=3000, seq_len=16, p=0.4, q=0.3):
    """
    获取重置Dyck语言数据集的便捷函数
    
    Args:
        n_samples: 样本数量
        seq_len: 序列长度
        p: 嵌套概率
        q: 连接概率
        
    Returns:
        RDyckDataset实例
    """
    vocab = create_rdyck_vocab()
    return RDyckDataset(n_samples, seq_len, p, q, vocab)


if __name__ == "__main__":
    # 测试Dyck数据集
    print("=== Dyck数据集测试 ===")
    dyck_dataset = get_dyck_dataset(n_samples=5, seq_len=12, num_pairs=2)
    
    for i in range(3):
        x, y = dyck_dataset[i]
        print(f"样本 {i+1}:")
        print(f"  输入张量: {x}")
        print(f"  输出张量: {y}")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {y.shape}")
        print()
    
    # 测试重置Dyck数据集
    print("=== 重置Dyck数据集测试 ===")
    rdyck_dataset = get_rdyck_dataset(n_samples=5, seq_len=15)
    
    for i in range(3):
        x, y = rdyck_dataset[i]
        print(f"样本 {i+1}:")
        print(f"  输入张量: {x}")
        print(f"  输出张量: {y}")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {y.shape}")
        print() 