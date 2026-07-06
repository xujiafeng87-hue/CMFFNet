import matplotlib.pyplot as plt

# ---------- 1. 读取 txt ----------
epochs = []
train_acc = []
val_acc = []
train_loss = []

with open('/mnt/ssd1/home/ailab/hlj/Passport/CNN_Mamba/Pro_log_5_10/MedMamba-main/epoch_metrics.txt', 'r') as f:
    next(f)  # 跳过表头
    for line in f:
        parts = line.strip().split()
        epochs.append(int(parts[0]))
        train_acc.append(float(parts[1]))
        train_loss.append(float(parts[2]))
        val_acc.append(float(parts[3]))

# ---------- 2. 保存 Train_Acc ----------
plt.figure()
plt.plot(epochs, train_acc, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Training Accuracy')
plt.grid(True)
plt.savefig('train_acc.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- 3. 保存 Val_Acc ----------
plt.figure()
plt.plot(epochs, val_acc, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy')
plt.grid(True)
plt.savefig('val_acc.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------- 4. 保存 Train_Loss ----------
plt.figure()
plt.plot(epochs, train_loss, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('train_loss.png', dpi=300, bbox_inches='tight')
plt.close()
