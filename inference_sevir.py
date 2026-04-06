import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import onnxruntime as ort


def load_onnx_model(onnx_path):
    """
    加载 ONNX 模型
    """
    try:
        # 配置 Session 选项，开启全面优化
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 限制为仅使用 CPU
        providers = ['CPUExecutionProvider']

        # 创建推理会话
        session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)

        print(f"--------------------------------ONNX 模型加载成功--------------------------------")
        print(f"模型路径: {onnx_path}")
        print(f"使用的执行提供器: {session.get_providers()}")
        return session
    except Exception as e:
        print(f"加载 ONNX 模型失败: {e}")
        return None


def get_sevir_input_data(data_path, sample_name):
    """
    获取 SEVIR 输入数据
    输出维度要求: (13, 1, 384, 384) -> (Seq_len, Channel, Height, Width)
    """
    print("正在加载 SEVIR 数据...")
    # 直接生成 Numpy 格式的 dummy data 代替
    data = np.load(os.path.join(data_path, sample_name), 'r')['sequence']
    input_data = np.expand_dims(data[:13].astype(np.float32), axis=1)
    target_data = np.expand_dims(data[13:].astype(np.float32), axis=1)
    return input_data, target_data


def predict(session, data_path, sample_name, saved_path):
    """
    使用 ONNX Runtime 进行单次正向推理
    """
    # 1. 获取输入数据: shape (13, 1, 384, 384); 输出数据: shape (12, 1, 384, 384)
    input_data, target_data = get_sevir_input_data(data_path, sample_name)

    # 2. 增加 Batch 维度: (13, 1, 384, 384) -> (1, 13, 1, 384, 384)
    input_data = np.expand_dims(input_data, axis=0)
    print(f"输入数据维度: {input_data.shape}, 真值数据维度: {target_data.shape}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 3. 开始推理
    # run 的第一个参数是你想获取的输出节点名称列表，第二个参数是输入字典
    output_data_list = session.run([output_name], {input_name: input_data / 255.})
    output_data = output_data_list[0]

    # 4. 后处理
    # 移除 Batch 维度: (1, 12, 1, 384, 384) -> (12, 1, 384, 384)
    output_numpy = np.squeeze(output_data, axis=0)
    output_numpy *= 255

    # 5. 结果可视化
    sample_plot_name = sample_name.split('.npz')[0] + '.png'
    save_pixel_image(input_data, output_numpy, target_data, saved_path, sample_plot_name)


def save_pixel_image(input_data, pred_img, target_img, saved_path, saved_name):
    """
    雷达模型外推结果
    """
    input_imgs = input_data[0, :, 0]
    pred_imgs = pred_img[:, 0]
    target_imgs = target_img[:, 0]

    # --- 1. 配置雷达回波色标 ---
    color_list = ['#4d4d4d', '#29be29', '#1a9618', '#0a690a', '#0b4b0a',
                  '#f5f502', '#edac00', '#f06f00', '#a00000', '#e700ff']
    cmap_radar = colors.ListedColormap(color_list)
    bounds = [16, 31, 52, 74, 103, 133, 160, 181, 219]
    norm_radar = colors.BoundaryNorm(bounds, cmap_radar.N, extend='both')

    row_labels = ["Input -> GT", "Prediction"]

    fig = plt.figure(figsize=(26, 12))
    gs = fig.add_gridspec(5, 13, width_ratios=[1] * 12 + [0.08], wspace=0.05, hspace=0.3)

    im_radar = None

    for t in range(12):
        ax = fig.add_subplot(gs[0, t])
        if t < 6:
            im_radar = ax.imshow(input_imgs[t + 6], cmap=cmap_radar, norm=norm_radar)
            if t == 5:
                ax.set_title(f"Input t=0", fontsize=10)
            else:
                ax.set_title(f"Input t-{5 - t}", fontsize=10)
        else:
            im_radar = ax.imshow(target_imgs[t - 6], cmap=cmap_radar, norm=norm_radar)
            ax.set_title(f"GT t+{t - 5}", fontsize=10)
        ax.axis('off')

        if t == 0:
            ax.text(-0.2, 0.5, row_labels[0], transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')

    for t in range(12):
        ax = fig.add_subplot(gs[1, t])
        if t < 6:
            ax.axis('off')  # 前6帧空白
        else:
            im_radar = ax.imshow(pred_imgs[t - 6], cmap=cmap_radar, norm=norm_radar)
            ax.set_title(f"Pred t+{t - 5}", fontsize=10)
            ax.axis('off')

        if t == 0:
            ax.text(-0.2, 0.5, row_labels[1], transform=ax.transAxes,
                    rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')

    cax_radar = fig.add_subplot(gs[0:2, 12])
    cb_radar = plt.colorbar(im_radar, cax=cax_radar, fraction=1.0, extend='both')
    cb_radar.set_label('Radar Reflectivity', fontsize=10)

    save_path = os.path.join(saved_path, saved_name)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"图片已保存至: {save_path}")


def main():
    # 1. 加载数据
    # SEVIR 原始数据的根目录
    base_sevir_path = './sevir_dataset'
    # 结果保存路径
    test_model_path = './'
    saved_path = os.path.join(test_model_path, 'predict_results')
    os.makedirs(saved_path, exist_ok=True)
    # SEVIR 测试数据的目录
    data_path = os.path.join(base_sevir_path, 'cascast/test/')
    # 样本名称
    sample_name = 'vil-2019-SEVIR_VIL_RANDOMEVENTS_2019_0501_0831.h5-0-0.npz'

    # 2. 加载 ONNX 模型
    # ONNX 模型文件路径
    onnx_path = os.path.join(test_model_path, 'swinlstm_model.onnx')
    session = load_onnx_model(onnx_path)

    # 3. 执行预测
    predict(session, data_path, sample_name, saved_path)


if __name__ == "__main__":
    main()
