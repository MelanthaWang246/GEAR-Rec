import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import datetime
from os.path import join
import register
from register import dataset
from sklearn.metrics.pairwise import cosine_similarity
import os

# ========== 初始化 ==========

utils.set_seed(world.seed)  # 固定随机种子，保证可复现
print(">>SEED:", world.seed)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 当前时间戳，格式如 20240409_153000
k = world.config['neighbor_k']  # 语义邻居数量，默认10

# 日志文件路径，包含数据集名、模型名、邻居数、时间戳，每次运行生成独立文件
ablation = world.config['ablation']
ablation_tag = f"_ablation_{ablation}" if ablation != 0 else ""
log_file = f"../logs/{world.dataset}_{world.model_name}_neighbor{str(k)}{ablation_tag}_{current_time}.txt"
_is_resume_log = False


# ========== 语义邻居构建 ========== → global

# 加载预训练的物品语义向量（由 LLM 生成，shape: (m_items, 1024)）
if world.item_semantic_emb_file:
    item_semantic_emb = torch.load(
        world.item_semantic_emb_file
    ).float()
else:
    item_semantic_emb = None
# 加载预训练的用户语义向量（shape: (n_users, 1024)）
if world.user_semantic_emb_file:
    user_semantic_emb = torch.load(
        world.user_semantic_emb_file
    ).float()
else:
    user_semantic_emb = None

# 计算所有物品两两之间的余弦相似度，得到 (m_items, m_items) 的相似度矩阵
cosine_sim_matrix = cosine_similarity(item_semantic_emb.numpy())

# 对每行（每个物品）按相似度降序排列，得到相似物品的索引排列
sorted_indices = np.argsort(-cosine_sim_matrix, axis=1)

# 取每个物品的 top-k 个最相似物品（第0列是自身，跳过）
# sorted_indices shape: (m_items, k)
sorted_indices = sorted_indices[:, 1:k+1]
sorted_indices = torch.tensor(sorted_indices).long()  # 转为 PyTorch 长整型张量


# ========== 模型构建 ==========

# 根据 world.model_name 从注册表中查找对应的模型类并实例化
# 对于 CoLaKG：传入语义邻居索引、物品语义向量、用户语义向量
Recmodel = register.MODELS[world.model_name](world.config, dataset, sorted_indices, item_semantic_emb, user_semantic_emb)
Recmodel = Recmodel.to(world.device)  # 将模型移到 GPU 或 CPU

# 构建 BPRLoss 对象，内部封装了 Adam 优化器
bpr = utils.BPRLoss(Recmodel, world.config) ## __init__()

# 生成模型权重的保存路径，如 checkpoints/colakg-ml-1m-3-64.pth.tar
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

LAST_CKPT_DIR = r"D:\Desktop\KGCN-colakg-semantic\code\last_checkpoints"
os.makedirs(LAST_CKPT_DIR, exist_ok=True)
last_weight_file = os.path.join(LAST_CKPT_DIR, os.path.basename(weight_file))

# ========== 早停初始状态 + 断点续训 ==========
start_epoch = 0
best_metric = -1.0
no_improve  = 0

if world.RESUME:
    try:
        ckpt = torch.load(last_weight_file, map_location=torch.device('cpu'))
        Recmodel.load_state_dict(ckpt['model_state_dict'])
        bpr.opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt['best_metric']
        no_improve  = ckpt['no_improve']
        _old_log    = ckpt.get('log_file', None)
        if _old_log and os.path.exists(_old_log):
            log_file       = _old_log
            _is_resume_log = True
        world.cprint(f"[RESUME] Loaded from {last_weight_file}, epoch {start_epoch}, best={best_metric:.6f}, no_improve={no_improve}")
    except FileNotFoundError:
        print(f"[RESUME] {last_weight_file} not found, starting from epoch 0")
    except KeyError as e:
        print(f"[RESUME] Checkpoint missing key {e}, starting from epoch 0")

os.makedirs(os.path.dirname(log_file), exist_ok=True)

# 如果 world.LOAD=1，尝试从文件加载已有权重（断点续训）；要么加载旧权重继续训练，要么保持当前初始化参数继续训练
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning") # 结束后继续下一行，不会中断

Neg_k = 1  # 每个正样本对应的负样本数量

# ========== 早停配置 ==========
EVAL_EVERY   = 5    # 每隔多少 epoch 做一次评估
PATIENCE     = 10   # 连续多少次评估无提升则停止（对应 EVAL_EVERY * PATIENCE 个 epoch）
MIN_DELTA    = 1e-5 # 提升小于此阈值视为无效（设为 0 则退化为严格大于）
MONITOR      = ('recall', -1)  # 监控指标：results[key][index]，-1 表示 topks 最后一个（最大 K）



# ========== TensorBoard 初始化 ==========

if world.tensorboard:
    # 创建 TensorBoard 写入器，日志目录包含时间戳和注释，便于区分多次实验
    w : SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")


# ========== 创建日志文件 ==========

train_start_time = datetime.datetime.now()
if _is_resume_log:
    with open(log_file, "a") as f:
        f.write(f"\n====================\n")
        f.write(f"[RESUME] Resumed at: {train_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[RESUME] Continuing from epoch {start_epoch} to {world.TRAIN_epochs}\n")
        f.write("====================\n")
else:
    with open(log_file, "w") as f:
        f.write("Training Log\n")
        f.write("====================\n")
        f.write(f"Start time: {train_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("====================\n")


# ========== 训练主循环 ==========

_interrupted = False  # 标记是否被中断

try:
    for epoch in range(start_epoch, world.TRAIN_epochs):
        start = time.time()

        # 每 EVAL_EVERY 个 epoch 进行一次评估
        if epoch % EVAL_EVERY == 0:
            cprint("[TEST]")
            test_results = Procedure.Test(dataset, Recmodel, epoch, w)
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_message = f'[{now}] TEST RESULTS at EPOCH[{epoch+1}/{world.TRAIN_epochs}]: {test_results}'
            print(log_message)
            with open(log_file, "a") as f:
                f.write(log_message + "\n")

            # 早停判断：监控 MONITOR 指标是否有提升
            metric_key, metric_idx = MONITOR
            current_metric = test_results[metric_key][metric_idx]
            if current_metric > best_metric + MIN_DELTA:
                best_metric = current_metric
                no_improve  = 0
                torch.save(Recmodel.state_dict(), weight_file)  # 只在最优时保存
                cprint(f"  [BEST] {metric_key}={best_metric:.6f}, model saved.")
            else:
                no_improve += 1
                cprint(f"  [no improve {no_improve}/{PATIENCE}] best {metric_key}={best_metric:.6f}")
                if no_improve >= PATIENCE:
                    msg = (f"[{now}] Early stopping triggered at EPOCH[{epoch+1}]: "
                           f"no improvement for {PATIENCE} evaluations "
                           f"({PATIENCE * EVAL_EVERY} epochs). "
                           f"Best {metric_key}={best_metric:.6f}")
                    print(msg)
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")
                    # break

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)

        end = time.time()
        epoch_time = end - start
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        log_message = f'[{now}] EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} - Time: {epoch_time:.2f} seconds'
        print(log_message)

        with open(log_file, "a") as f:
            f.write(log_message + "\n")

except KeyboardInterrupt:
    _interrupted = True
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f'[{now}] *** Training interrupted by user (KeyboardInterrupt) at EPOCH[{epoch+1}/{world.TRAIN_epochs}] ***'
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

except Exception as e:
    _interrupted = True
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    import traceback
    tb = traceback.format_exc()
    msg = f'[{now}] *** Training stopped due to exception at EPOCH[{epoch+1}/{world.TRAIN_epochs}] ***\n{tb}'
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    raise

finally:
    train_end_time = datetime.datetime.now()
    elapsed = train_end_time - train_start_time
    status = "INTERRUPTED" if _interrupted else "COMPLETED"
    summary = (
        f"====================\n"
        f"End time : {train_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Elapsed  : {str(elapsed).split('.')[0]}\n"
        f"Status   : {status}\n"
        f"===================="
    )
    print(summary)
    with open(log_file, "a") as f:
        f.write(summary + "\n")

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': Recmodel.state_dict(),
            'optimizer_state_dict': bpr.opt.state_dict(),
            'best_metric': best_metric,
            'no_improve': no_improve,
            'log_file': log_file,
        }, last_weight_file)
        print(f"Last epoch model saved to {last_weight_file}")
    except Exception:
        pass

    if world.tensorboard:
        w.close()
