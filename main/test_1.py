import json
import os
import time

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torchvision import transforms

import data_input
from MedMamba import ConvMamba, MambaEncoder, VSSM as medmamba


MODEL_WEIGHT_PATH = "/mnt/ssd1/home/ailab/hlj/Passport/MedMamba/Pro_log_5_10/main/best_model.pth"
DATA_ROOT = "/home/ailab/hlj/Passport/数据集2025-12/"
EVALUATION_TXT = "evaluation5_27.txt"
PRED_LABELS_TXT = "pre_labels.txt"


class_indict = {}


def load_class_indices(path="./class_indices.json"):
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


data_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model():
    return medmamba(
        depths=[2, 2, 4, 2],
        dims=[96, 192, 384, 768],
        num_classes=4,
    )


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _save_ops(module, ops):
    module.total_ops = torch.tensor(float(ops), dtype=torch.float64)


def _count_conv(module, inputs, output):
    out = _first_tensor(output)
    if out is None:
        return

    kernel_ops = module.in_channels // module.groups
    for kernel_size in module.kernel_size:
        kernel_ops *= kernel_size
    bias_ops = 1 if module.bias is not None else 0
    _save_ops(module, out.numel() * (kernel_ops + bias_ops))


def _count_linear(module, inputs, output):
    out = _first_tensor(output)
    if out is None:
        return

    bias_ops = 1 if module.bias is not None else 0
    _save_ops(module, out.numel() * (module.in_features + bias_ops))


def _count_norm(module, inputs, output):
    out = _first_tensor(output)
    if out is None:
        return

    if isinstance(module, nn.LayerNorm):
        _save_ops(module, out.numel() * 5)
    else:
        _save_ops(module, out.numel() * 2)


def _count_activation(module, inputs, output):
    out = _first_tensor(output)
    if out is None:
        return

    _save_ops(module, out.numel())


def _count_pool(module, inputs, output):
    inp = _first_tensor(inputs[0]) if inputs else None
    out = _first_tensor(output)
    if inp is None or out is None:
        return

    if isinstance(
        module,
        (
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool3d,
        ),
    ):
        kernel_ops = max(1, inp.numel() // max(1, out.numel()))
    else:
        kernel = module.kernel_size
        if isinstance(kernel, int):
            kernel_ops = kernel
        else:
            kernel_ops = 1
            for kernel_size in kernel:
                kernel_ops *= kernel_size

    _save_ops(module, out.numel() * kernel_ops)


def _count_multihead_attention(module, inputs, output):
    query = _first_tensor(inputs[0]) if len(inputs) > 0 else None
    key = _first_tensor(inputs[1]) if len(inputs) > 1 else query
    value = _first_tensor(inputs[2]) if len(inputs) > 2 else key
    out = _first_tensor(output)
    if query is None or key is None or value is None or out is None:
        return

    if getattr(module, "batch_first", False):
        batch_size, target_len, embed_dim = query.shape[:3]
        source_len = key.shape[1]
        value_len = value.shape[1]
    else:
        target_len, batch_size, embed_dim = query.shape[:3]
        source_len = key.shape[0]
        value_len = value.shape[0]

    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    qkv_proj_ops = batch_size * (target_len + source_len + value_len) * embed_dim * embed_dim
    attn_scores_ops = batch_size * num_heads * target_len * source_len * head_dim
    attn_weighted_value_ops = batch_size * num_heads * target_len * source_len * head_dim
    out_proj_ops = out.numel() * embed_dim
    _save_ops(module, qkv_proj_ops + attn_scores_ops + attn_weighted_value_ops + out_proj_ops)


def _count_mamba_encoder_extra(module, inputs, output):
    inp = _first_tensor(inputs[0]) if inputs else None
    if inp is None:
        return

    batch_size, sequence_len, dim = inp.shape
    mamba = module.mamba
    d_state = int(getattr(mamba, "d_state", 16))
    d_conv = int(getattr(mamba, "d_conv", 4))
    expand = int(getattr(mamba, "expand", 2))
    d_inner = int(getattr(mamba, "d_inner", dim * expand))
    dt_rank = getattr(mamba, "dt_rank", None)
    if dt_rank is None or dt_rank == "auto":
        dt_rank = max(1, (dim + 15) // 16)
    dt_rank = int(dt_rank)

    # mamba_ssm.Mamba uses fused/custom ops that are not reliably visible to
    # standard nn.Linear/Conv1d hooks. Count the major pieces explicitly:
    # in projection, causal depthwise conv, x/dt projections, selective scan,
    # output projection, gate/activation work, and the encoder residual add.
    in_proj_ops = batch_size * sequence_len * dim * (2 * d_inner)
    depthwise_conv_ops = batch_size * sequence_len * d_inner * d_conv
    x_proj_ops = batch_size * sequence_len * d_inner * (dt_rank + 2 * d_state)
    dt_proj_ops = batch_size * sequence_len * dt_rank * d_inner
    selective_scan_ops = batch_size * sequence_len * d_inner * d_state * 9
    out_proj_ops = batch_size * sequence_len * d_inner * dim
    gate_and_activation_ops = batch_size * sequence_len * d_inner * 4
    residual_add_ops = inp.numel()
    _save_ops(
        module,
        in_proj_ops
        + depthwise_conv_ops
        + x_proj_ops
        + dt_proj_ops
        + selective_scan_ops
        + out_proj_ops
        + gate_and_activation_ops
        + residual_add_ops,
    )


def _count_convmamba_extra(module, inputs, output):
    inp = _first_tensor(inputs[0]) if inputs else None
    if inp is None:
        return

    # Counts functional ops not represented as child modules: block residual
    # additions, functional ReLUs, and final concatenation/projection plumbing.
    batch_size, token_count, dim = inp.shape
    block_residual_adds = inp.numel() * module.depth
    inner_relu_ops = batch_size * token_count * dim * module.depth
    center_relu_ops = batch_size * dim
    concat_ops = batch_size * dim * 2
    _save_ops(module, block_residual_adds + inner_relu_ops + center_relu_ops + concat_ops)


def count_parameters_and_flops(model, input_size=(1, 3, 128, 128)):
    """
    Count parameters and estimate FLOPs with forward hooks.

    Convention: 1 MAC is counted as 1 FLOP. Conv/Linear, norms, activations,
    pooling, MultiheadAttention and ConvMamba/Mamba fused ops are covered.
    mamba_ssm fused kernels are approximate in this hook-based estimate.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    param_size_mb = total_params * 4 / (1024**2)
    module_param_summary = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }

    for module in model.modules():
        if hasattr(module, "total_ops"):
            delattr(module, "total_ops")

    hook_map = (
        ((nn.Conv1d, nn.Conv2d, nn.Conv3d), _count_conv),
        (nn.Linear, _count_linear),
        (
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
            _count_norm,
        ),
        ((nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Softmax), _count_activation),
        (
            (
                nn.MaxPool1d,
                nn.MaxPool2d,
                nn.MaxPool3d,
                nn.AvgPool1d,
                nn.AvgPool2d,
                nn.AvgPool3d,
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveAvgPool3d,
                nn.AdaptiveMaxPool1d,
                nn.AdaptiveMaxPool2d,
                nn.AdaptiveMaxPool3d,
            ),
            _count_pool,
        ),
        (nn.MultiheadAttention, _count_multihead_attention),
        (MambaEncoder, _count_mamba_encoder_extra),
        (ConvMamba, _count_convmamba_extra),
    )

    hooks = []
    for module in model.modules():
        for module_types, hook_fn in hook_map:
            if isinstance(module, module_types):
                hooks.append(module.register_forward_hook(hook_fn))
                break

    flops = 0
    module_flop_summary = {name: 0 for name in module_param_summary}
    was_training = model.training
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size, device=device)

        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)

        flops = sum(module.total_ops.item() for module in model.modules() if hasattr(module, "total_ops"))
        module_flop_summary = {}
        for name, child in model.named_children():
            module_flop_summary[name] = sum(
                module.total_ops.item() for module in child.modules() if hasattr(module, "total_ops")
            )

        print(f"\n{'=' * 60}")
        print("Model complexity")
        print(f"{'=' * 60}")
        print(f"Input size: {tuple(input_size)}")
        print("FLOPs convention: 1 MAC = 1 FLOP")
        print("FLOPs note: mamba_ssm fused kernels are approximate but include projection/conv/scan estimates.")
        print(f"Total parameters: {total_params / 1e6:.2f}M ({total_params:,})")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params:,})")
        print(f"Non-trainable parameters: {non_trainable_params / 1e6:.2f}M ({non_trainable_params:,})")
        print(f"Parameter size: {param_size_mb:.2f} MB")
        print(f"FLOPs: {flops / 1e9:.2f}G ({int(flops):,})")
        print("\nTop-level module breakdown:")
        for name in module_param_summary:
            module_params = module_param_summary[name]
            module_flops = module_flop_summary.get(name, 0)
            print(f"  {name}: params={module_params / 1e6:.2f}M, FLOPs={module_flops / 1e9:.2f}G")
        print(f"{'=' * 60}\n")
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        print("Skipping FLOPs calculation.")
        flops = 0
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return (
        total_params,
        trainable_params,
        non_trainable_params,
        flops,
        param_size_mb,
        module_param_summary,
        module_flop_summary,
    )


def get_model_config_summary(model):
    stages = getattr(model, "stages", None)
    if stages is None:
        local_trans = getattr(model, "local_trans_pixel", None)
        stages = [local_trans] if local_trans is not None else []
    if not stages:
        return {}

    first_stage = stages[0]
    first_layer = first_stage.layers[0] if len(first_stage.layers) > 0 else None
    mamba = getattr(first_layer, "mamba", None) if first_layer is not None else None
    return {
        "input_size": getattr(model, "input_size", None),
        "patch_size": getattr(model, "patch_size", None),
        "initial_grid_size": getattr(model, "grid_size", None),
        "embed_dim": getattr(model, "embed_dim", None),
        "num_features": getattr(model, "num_features", None),
        "dims": getattr(model, "dims", None),
        "depths": getattr(model, "depths", None),
        "stage_grid_sizes": [getattr(stage, "grid_size", None) for stage in stages],
        "stage_token_counts": [
            getattr(stage, "grid_size", 0) ** 2 if getattr(stage, "grid_size", None) else None for stage in stages
        ],
        "d_state": getattr(mamba, "d_state", None),
        "d_conv": getattr(mamba, "d_conv", None),
        "expand": getattr(mamba, "expand", None),
    }


def get_model(model_weight=MODEL_WEIGHT_PATH):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print("测试设备：", device)
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    model.eval()
    return model, device


def get_inputs(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = data_transform(image)
    inputs = torch.unsqueeze(inputs, dim=0)
    return inputs


def predict(model, inputs, device):
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        max_value, index = torch.max(outputs, dim=1)

    return index, max_value


def main(model, device, img_path="./1.png"):
    global class_indict
    if not class_indict and os.path.exists("./class_indices.json"):
        class_indict = load_class_indices()

    inputs = get_inputs(img_path)
    index, max_value = predict(model, inputs, device)
    pred_index = index.cpu().item()
    return pred_index, class_indict.get(str(pred_index), str(pred_index)), max_value.cpu().item()


def pre_write_txt(pred, file):
    for i in pred:
        with open(file, "a", encoding="utf-8_sig") as f:
            f.write(str(i) + ",")
    print("-----------------预测结果已经写入文本文件--------------------")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warmup_model(model, device, warmup_iters=10):
    if device.type != "cuda":
        return

    print(f"Performing CUDA warmup ({warmup_iters} forward passes)...")
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    torch.cuda.synchronize(device)
    print("CUDA warmup completed\n")


def test_acc(model, device):
    global class_indict
    if not class_indict:
        class_indict = load_class_indices()

    acc = 0.0
    model.to(device)
    model.eval()
    batch_index = 0
    all_pred = []
    all_label = []
    total_inference_time = 0.0
    num_samples = 0

    (
        total_params,
        trainable_params,
        non_trainable_params,
        flops,
        param_size_mb,
        module_param_summary,
        module_flop_summary,
    ) = count_parameters_and_flops(model, input_size=(1, 3, 128, 128))
    model_config_summary = get_model_config_summary(model)
    _warmup_model(model, device, warmup_iters=10)

    print("Starting inference on test set...\n")
    with torch.no_grad():
        for data_test in validate_loader:
            batch_index += 1
            test_images, test_labels = data_test
            test_labels_len = len(test_labels)
            num_samples += test_labels_len

            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            _synchronize(device)
            start_time = time.perf_counter()
            outputs = model(test_images)
            _synchronize(device)
            total_inference_time += time.perf_counter() - start_time

            _, predict_y = torch.max(outputs, dim=1)
            val_acc = (predict_y == test_labels).sum().item()
            acc += val_acc

            if batch_index % 10 == 0 or batch_index == 1:
                print(f"Batch {batch_index}: acc={val_acc / test_labels_len:.4f} ({val_acc}/{test_labels_len})")

            all_pred.extend(predict_y.cpu().numpy().tolist())
            all_label.extend(test_labels.cpu().numpy().tolist())

    accurate_test = acc / num_samples if num_samples > 0 else 0
    avg_inference_time = total_inference_time / num_samples if num_samples > 0 else 0
    inference_speed = 1 / avg_inference_time if avg_inference_time > 0 else 0
    time_note = "Inference time is forward-only; data loading and host-to-device transfer are excluded."
    flops_note = (
        "FLOPs convention: 1 MAC = 1 FLOP; mamba_ssm fused kernels are approximate "
        "but include projection/conv/scan estimates."
    )

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"测试集样本数: {num_samples}")
    print(f"批次数: {batch_index}")
    print(f"Batch size: {validate_loader.batch_size}")
    print(f"\nAccuracy: {accurate_test:.4f} ({accurate_test * 100:.2f}%)")
    print("\n推理时间统计:")
    print(f"  Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"  Average Inference Time per sample: {avg_inference_time:.6f} seconds")
    print(f"  Inference Speed: {inference_speed:.2f} samples/second")
    print(f"  Inference Speed: {inference_speed * 60:.2f} samples/minute")
    print(f"  {time_note}")
    print("\n模型复杂度:")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"  Non-trainable parameters: {non_trainable_params / 1e6:.2f}M")
    print(f"  Parameter size: {param_size_mb:.2f} MB")
    print(f"  FLOPs: {flops / 1e9:.2f}G")
    print(f"  {flops_note}")
    print("\n  模型配置:")
    for name, value in model_config_summary.items():
        print(f"    {name}: {value}")
    print("\n  模块分解:")
    for name in module_param_summary:
        print(
            f"    {name}: params={module_param_summary[name] / 1e6:.2f}M, "
            f"FLOPs={module_flop_summary.get(name, 0) / 1e9:.2f}G"
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        all_label, all_pred, average="weighted", zero_division=0
    )

    class_names = [class_indict[str(i)] for i in range(len(class_indict))]
    report = classification_report(
        all_label,
        all_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("分类报告 (Classification Report)")
    print("=" * 60)
    print(report)

    with open(EVALUATION_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CNN_ConvMamba/ConvMamba 模型评估报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("一、性能指标\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {accurate_test:.4f} ({accurate_test * 100:.2f}%)\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1 Score (weighted): {f1:.4f}\n\n")

        f.write("二、推理时间统计\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total Inference Time: {total_inference_time:.4f} seconds\n")
        f.write(f"Average Inference Time per sample: {avg_inference_time:.6f} seconds\n")
        f.write(f"Inference Speed: {inference_speed:.2f} samples/second\n")
        f.write(f"Inference Speed: {inference_speed * 60:.2f} samples/minute\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Batch size: {validate_loader.batch_size}\n")
        f.write(f"Number of batches: {batch_index}\n")
        f.write(time_note + "\n\n")

        f.write("三、模型复杂度\n")
        f.write("-" * 60 + "\n")
        f.write(f"Total parameters: {total_params / 1e6:.2f}M ({total_params:,})\n")
        f.write(f"Trainable parameters: {trainable_params / 1e6:.2f}M ({trainable_params:,})\n")
        f.write(f"Non-trainable parameters: {non_trainable_params / 1e6:.2f}M ({non_trainable_params:,})\n")
        f.write(f"Parameter size: {param_size_mb:.2f} MB\n")
        f.write(f"FLOPs: {flops / 1e9:.2f}G ({int(flops):,})\n")
        f.write(flops_note + "\n\n")

        f.write("模型配置\n")
        f.write("-" * 60 + "\n")
        for name, value in model_config_summary.items():
            f.write(f"{name}: {value}\n")
        f.write("\n")

        f.write("模块分解\n")
        f.write("-" * 60 + "\n")
        for name in module_param_summary:
            f.write(
                f"{name}: params={module_param_summary[name] / 1e6:.2f}M "
                f"({module_param_summary[name]:,}), "
                f"FLOPs={module_flop_summary.get(name, 0) / 1e9:.2f}G "
                f"({int(module_flop_summary.get(name, 0)):,})\n"
            )
        f.write("\n")

        f.write("四、分类报告\n")
        f.write("-" * 60 + "\n")
        f.write(report)
        f.write("\n\n")

        if support is not None:
            f.write("五、各类别样本数\n")
            f.write("-" * 60 + "\n")
            for i, name in enumerate(class_names):
                f.write(f"{name}: {support[i]} samples\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("评估完成\n")
        f.write("=" * 60 + "\n")

    with open(PRED_LABELS_TXT, "w", encoding="utf-8") as f:
        f.write("prediction true_label\n")
        for pred, label in zip(all_pred, all_label):
            f.write(f"{pred} {label}\n")

    print("\n" + "=" * 60)
    print("文件保存成功")
    print("=" * 60)
    print(f"已将评估结果写入: {EVALUATION_TXT}")
    print(f"已将预测标签写入: {PRED_LABELS_TXT}")


def testset_acc_one_by_one(test_dir=None):
    global class_indict
    if test_dir is None:
        test_dir = os.path.join(DATA_ROOT, "val")

    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' does not exist!")
        return

    if not class_indict:
        if os.path.exists("./class_indices.json"):
            class_indict = load_class_indices()
        else:
            class_names = sorted(name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name)))
            class_indict = {str(index): name for index, name in enumerate(class_names)}

    class_list = os.listdir(test_dir)
    print(f"\n开始逐类别测试，共 {len(class_list)} 个类别\n")

    total_correct = 0
    total_samples = 0
    model, device = get_model()

    for class_dir in class_list:
        class_path = os.path.join(test_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        correct_num = 0
        img_list = os.listdir(class_path)
        total_in_class = len(img_list)

        for img in img_list:
            img_path = os.path.join(class_path, img)
            pred_idx, pred_label, confidence = main(model, device, img_path)
            if pred_label == class_dir:
                correct_num += 1

        total_correct += correct_num
        total_samples += total_in_class
        accuracy = correct_num / total_in_class if total_in_class > 0 else 0
        print(f"{class_dir:20s}: {correct_num:5d}/{total_in_class:5d}, accuracy={accuracy:.4f}")

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"总体准确率: {total_correct}/{total_samples} = {overall_accuracy:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    model, device = get_model(model_weight=MODEL_WEIGHT_PATH)

    validate_loader, val_num = data_input.ImageFolder_dataloader(
        root_dir=DATA_ROOT,
        bs=2,
        is_train=False,
        nw=4,
    )
    class_indict = load_class_indices()
    test_acc(model, device)
