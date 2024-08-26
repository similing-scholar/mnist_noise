import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from PIL import Image

# 加载MNIST数据集
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST('../dataset', train=False, download=False, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

# 提取指定数字的图像
def extract_digit_images(digit, data_loader, start_index=0):
    digit_images = []
    current_index = 0
    for image, label in data_loader:
        if label.item() == digit:
            if current_index >= start_index:
                digit_images.append(image)
            current_index += 1
            if len(digit_images) >= 1:
                break
    return digit_images[0].squeeze().numpy() if digit_images else None, current_index

# 模拟脉冲亮度
def simulate_luminance(pulse_count, pulse_duration, pulse_interval, A, tau, decay_tau, decay_factor_initial,
                       decay_factor_reduction):
    total_time = pulse_interval * (pulse_count - 1) + pulse_duration
    time = np.linspace(0, total_time, total_time)
    luminance = np.zeros_like(time)

    previous_end_luminance = 0
    decay_factor = decay_factor_initial
    for i in range(pulse_count):
        start_time = i * pulse_interval
        end_time = start_time + pulse_duration
        pulse_time = np.linspace(0, pulse_duration, pulse_duration)

        current_luminance = A * (1 - np.exp(-pulse_time / tau)) + previous_end_luminance
        luminance[start_time:end_time] = current_luminance

        if i < pulse_count - 1:
            next_start_time = (i + 1) * pulse_interval
            forgetting_time = np.linspace(0, pulse_interval - pulse_duration, pulse_interval - pulse_duration)
            forgetting_curve = current_luminance[-1] * np.exp(-forgetting_time / decay_tau)
            luminance[end_time:next_start_time] = forgetting_curve

        previous_end_luminance = luminance[next_start_time - 1] if i < pulse_count - 1 else 0
        decay_factor = max(0, decay_factor - decay_factor_reduction)

    return time, luminance

# 生成噪声图像
def generate_noise_image(noise_level, noise_type, shape=(28, 28), logo_path=None, logo_alpha=0.5):
    noise_image = np.zeros(shape)
    if noise_type == 'percent高斯噪声':
        noise_image = np.random.normal(0, noise_level, shape)
    elif noise_type == '网格噪声':
        grid_value = np.random.normal(0, noise_level)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if (i + j) % 2 == 0:
                    noise_image[i, j] = grid_value
    elif noise_type == '条纹噪声':
        noise_image = _generate_striped_noise(shape, noise_level)
    elif noise_type == '马赛克噪声':
        noise_image = _generate_mosaic_noise(shape, noise_level)
    elif noise_type == 'logo噪声' and logo_path:
        noise_image = _generate_logo_noise(shape, logo_path, logo_alpha)
    return noise_image

# 生成条纹噪声
def _generate_striped_noise(size, noise_level):
    noise = np.zeros(size)
    stripe_width = 2
    for i in range(0, size[1], stripe_width * 2):
        noise[:, i:i+stripe_width] = noise_level
    return noise

# 生成马赛克噪声
def _generate_mosaic_noise(size, noise_level):
    noise = np.random.rand(*size)
    mosaic_size = 4
    for i in range(0, size[0], mosaic_size):
        for j in range(0, size[1], mosaic_size):
            block_value = np.random.normal(0, noise_level)
            noise[i:i+mosaic_size, j:j+mosaic_size] = block_value
    return noise

# 生成logo噪声
def _generate_logo_noise(size, logo_path, logo_alpha):
    logo = Image.open(logo_path).convert("L")
    logo = logo.resize((size[1], size[0]))
    logo_array = np.array(logo) / 255.0  # 将logo灰度值归一化到[0, 1]
    return logo_array * logo_alpha  # 将logo灰度值乘以透明度因子

# 生成28x28的像素矩阵
def generate_pixel_matrix(digit_image, luminance, chosen_time_index, noise_image):
    pixel_matrix = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            if digit_image[i, j] > 0.3:
                pixel_matrix[i, j] = luminance[chosen_time_index]

    # 将噪声与像素矩阵叠加
    pixel_matrix = pixel_matrix + noise_image

    # 将值限制在0-1之间
    pixel_matrix = np.clip(pixel_matrix, 0, 1)

    return pixel_matrix

# 绘制亮度变化曲线
def plot_luminance_curve(time, luminance):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, luminance, 'o-')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Luminance (a. u.)')
    ax.set_title('Luminance vs Time under Pulse Voltage with Sliding Window Effect')
    ax.grid(True)
    return fig

# 绘制28x28像素矩阵图像
def plot_pixel_matrix(pixel_matrix):
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.imshow(pixel_matrix, cmap='gray', interpolation='nearest')
    fig.colorbar(cax, label='Luminance (a. u.)')
    ax.set_title('28x28 Pixel Matrix with Luminance and Noise')
    return fig

# Streamlit 页面设置
st.set_page_config(layout="wide")
st.title('MNIST 数字脉冲亮度模拟')

# 选择输入的数字
digit = st.sidebar.selectbox('选择数字', list(range(10)))

# 加载数据
data_loader = load_data()
start_index = st.session_state.get('start_index', 0)

# 提取图像
digit_image, current_index = extract_digit_images(digit, data_loader, start_index)
if digit_image is not None:
    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

    # 模拟参数
    pulse_count = col1.number_input('脉冲数量', min_value=1, value=6)
    pulse_duration = col1.number_input('每个脉冲持续时间/亮度上升 (ms)', min_value=1, value=500)
    pulse_interval = col1.number_input('单个亮度上升下降区间时间 (ms)', min_value=pulse_duration, value=800)
    A = col1.number_input('最大亮度', min_value=1, value=10)
    tau = col1.number_input('上升时间常数', min_value=1, value=200)
    decay_tau = col1.number_input('衰减时间常数', min_value=1, value=1000)
    decay_factor_initial = col1.number_input('初始衰减比例', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    decay_factor_reduction = col1.number_input('每次脉冲衰减减少量', min_value=0.01, max_value=0.1, value=0.05, step=0.01)

    # 亮度模拟
    time, luminance = simulate_luminance(pulse_count, pulse_duration, pulse_interval, A, tau, decay_tau,
                                         decay_factor_initial, decay_factor_reduction)

    # 生成噪声图像
    chosen_time_index = col2.slider('选择时间点', min_value=0, max_value=int(time[-1]), value=int(time[-1] // 4))
    noise_level = col2.number_input('噪声强度', min_value=0, value=2)
    noise_type = col2.selectbox('选择噪声类型', ['percent高斯噪声', '网格噪声', '条纹噪声', '马赛克噪声', 'logo噪声'])
    logo_path = 'D:/BITcode/__code__/PJ_LML/dataset/BIT.jpg' if noise_type == 'logo噪声' else None
    logo_alpha = col2.slider('Logo透明度', min_value=0.0, max_value=1.0, value=0.5) if noise_type == 'logo噪声' else 0.5
    noise_image = generate_noise_image(noise_level, noise_type, logo_path=logo_path, logo_alpha=logo_alpha)

    # 生成像素矩阵
    pixel_matrix = generate_pixel_matrix(digit_image, luminance, chosen_time_index, noise_image)

    # 绘制图像
    luminance_fig = plot_luminance_curve(time, luminance)
    col3.pyplot(luminance_fig)

    pixel_matrix_fig = plot_pixel_matrix(pixel_matrix)
    col3.pyplot(pixel_matrix_fig)

    # 下一张图片按钮
    if col2.button('切换至下一张图片'):
        st.session_state['start_index'] = current_index
        st.experimental_rerun()

    # 文件保存地址
    save_path = col2.text_input('保存文件夹的路径', 'C:\\Users\\JiaPeng\\Desktop\\test')

    # 获取当前时间
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    # 保存按钮
    if col2.button('保存亮度曲线图片'):
        luminance_fig.savefig(os.path.join(save_path, f'luminance_curve_{current_time}.png'), dpi=300)
        st.success(f'亮度曲线图片已保存到 {save_path}')

    if col2.button('保存矩阵图像图片'):
        pixel_matrix_fig.savefig(os.path.join(save_path, f'pixel_matrix_{current_time}.png'), dpi=300)
        st.success(f'矩阵图像图片已保存到 {save_path}')

    if col2.button('保存原始数据'):
        luminance_data = np.column_stack((time, luminance))
        pixel_matrix_data = pixel_matrix

        # 保存为Excel文件
        with pd.ExcelWriter(os.path.join(save_path, f'data_{current_time}.xlsx')) as writer:
            pd.DataFrame(luminance_data, columns=['Time (ms)', 'Luminance (a. u.)']).to_excel(writer, sheet_name='Luminance Data', index=False)
            pd.DataFrame(pixel_matrix_data).to_excel(writer, sheet_name='Pixel Matrix Data', index=False)
            params = {
                '脉冲数量': [pulse_count],
                '每个脉冲持续时间/亮度上升 (ms)': [pulse_duration],
                '单个亮度上升下降区间时间 (ms)': [pulse_interval],
                '最大亮度': [A],
                '上升时间常数': [tau],
                '衰减时间常数': [decay_tau],
                '初始衰减比例': [decay_factor_initial],
                '每次脉冲衰减减少量': [decay_factor_reduction],
                '噪声强度': [noise_level]
            }
            pd.DataFrame(params).to_excel(writer, sheet_name='Parameters', index=False)

        st.success(f'原始数据已保存到 {save_path}')
else:
    st.warning('未找到该数字的图像，请选择其他数字。')
