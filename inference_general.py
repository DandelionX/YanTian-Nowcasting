import os
import numpy as np
from datetime import datetime
import onnxruntime as ort
from scipy.ndimage import zoom


def load_onnx_model(onnx_path):
    """
    加载 ONNX 模型
    """
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)

        print(f"--------------------------------ONNX 模型加载成功--------------------------------")
        return session
    except Exception as e:
        print(f"加载 ONNX 模型失败: {e}")
        return None


def preprocess_sequence(input_array, target_seq_len=13, target_size=(384, 384)):
    """
    通用数据预处理函数
    输入:
        input_array: 任意形状的 numpy 数组，shape 为 (T, H, W)
    输出:
        processed_data: 处理后符合模型输入的数组，shape 为 (13, 1, 384, 384)
        orig_hw: 记录原始的 (H, W) 用于后处理还原
    """
    T, H, W = input_array.shape
    orig_hw = (H, W)

    # 1. 处理时间维度 (T -> 13)
    if T < target_seq_len:
        # 如果帧数不够，在最前面用 0 填充 (代表历史时刻无雷达回波)
        pad_width = ((target_seq_len - T, 0), (0, 0), (0, 0))
        data_padded = np.pad(input_array, pad_width, mode='constant', constant_values=0)
    elif T > target_seq_len:
        # 如果帧数过多，只取最近的 target_seq_len 帧
        data_padded = input_array[-target_seq_len:]
    else:
        data_padded = input_array

    # 2. 处理空间维度 (H, W -> 384, 384)
    # 计算缩放比例，时间维度不变(1.0)
    zoom_factors = (1.0, target_size[0] / H, target_size[1] / W)

    # order=1 表示使用双线性插值 (Bilinear interpolation)
    data_resized = zoom(data_padded, zoom_factors, order=1)

    # 3. 增加 Channel 维度 (13, 384, 384) -> (13, 1, 384, 384)
    processed_data = np.expand_dims(data_resized, axis=1).astype(np.float32)

    return processed_data, orig_hw


def postprocess_sequence(output_array, orig_hw):
    """
    通用数据后处理函数
    输入:
        output_array: 模型输出的数组，shape 为 (12, 1, 384, 384)
        orig_hw: 原始的空间分辨率 (H, W)
    输出:
        restored_data: 还原分辨率后的数组，shape 为 (12, 1, H, W)
    """
    # 移除 Channel 维度以便于插值: (12, 1, 384, 384) -> (12, 384, 384)
    output_squeezed = np.squeeze(output_array, axis=1)

    _, H_curr, W_curr = output_squeezed.shape
    target_H, target_W = orig_hw

    # 计算还原的缩放比例
    zoom_factors = (1.0, target_H / H_curr, target_W / W_curr)

    # 插值还原到原始分辨率
    data_restored = zoom(output_squeezed, zoom_factors, order=1)

    # 加回 Channel 维度: (12, H, W) -> (12, 1, H, W)
    final_output = np.expand_dims(data_restored, axis=1).astype(np.float32)

    return final_output


def predict_general(session, input_array, save_dir, task_name="random_task"):
    """
    通用推理全流程
    """
    print(f"\n[任务: {task_name}] 开始执行...")
    print(f"1. 接收到原始数据，维度: {input_array.shape}")

    # ================= 预处理 =================
    model_input, orig_hw = preprocess_sequence(input_array)
    # 增加 Batch 维度: (13, 1, 384, 384) -> (1, 13, 1, 384, 384)
    model_input = np.expand_dims(model_input, axis=0)
    print(f"2. 预处理完成，输入模型的数据维度: {model_input.shape}")

    # ================= 推理 =================
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 归一化输入到 0-1
    output_data_list = session.run([output_name], {input_name: model_input / 255.0})
    output_data = output_data_list[0]

    # 移除 Batch 维度，并还原数值范围: (1, 12, 1, 384, 384) -> (12, 1, 384, 384)
    output_numpy = np.squeeze(output_data, axis=0) * 255.0

    # ================= 后处理 =================
    final_result = postprocess_sequence(output_numpy, orig_hw)
    print(f"3. 模型推理与后处理完成，最终预报结果维度: {final_result.shape}")

    # ================= 保存结果 =================
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{task_name}_pred.npy")
    np.save(save_path, final_result)
    print(f"4. 结果已保存至: {save_path}")

    return final_result


def main():
    # 1. 模型路径准备
    test_model_path = './'
    onnx_path = os.path.join(test_model_path, 'swinlstm_model.onnx')
    save_dir = os.path.join(test_model_path, 'predict_results_general')

    # 2. 加载模型
    session = load_onnx_model(onnx_path)

    # 3. 模拟场景：接收到一个 (10时次, 200宽, 200高) 的随机数据*******（已做好归一化）*******
    # 这里乘以 255 是为了模拟真实的雷达回波数值范围 (0~255)
    dummy_input_10_200 = np.random.rand(10, 200, 200).astype(np.float32) * 255.0

    # 4. 执行通用预测
    predict_general(session, input_array=dummy_input_10_200, save_dir=save_dir, task_name="test_10x200x200")


if __name__ == "__main__":
    main()