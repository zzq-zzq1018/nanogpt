import os
import numpy as np
import tiktoken

# 配置参数
input_file_path = "input.txt"  # 你的原始诗歌文本文件，放在 data/poemtext/ 下
encoding = tiktoken.get_encoding("gpt2")  # 字符级/词级编码，和训练配置保持一致
val_fraction = 0.1  # 验证集占比 10%

def main():
    # 读取原始文本
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    # 编码文本
    ids = encoding.encode(data)
    print(f"总 token 数: {len(ids)}")

    # 划分训练集和验证集
    split_idx = int(len(ids) * (1 - val_fraction))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    # 保存为二进制文件
    train_ids_np = np.array(train_ids, dtype=np.uint16)
    val_ids_np = np.array(val_ids, dtype=np.uint16)

    train_ids_np.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_ids_np.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    print(f"训练集大小: {len(train_ids)} tokens")
    print(f"验证集大小: {len(val_ids)} tokens")
    print("数据预处理完成，已生成 train.bin 和 val.bin")

if __name__ == "__main__":
    main()