# Enhanced ZSSR v3 with Easy Hyperparameter Configuration - FIXED
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import tqdm
import random
import matplotlib.pyplot as plt
from collections import deque
from torch.cuda.amp import autocast, GradScaler
import os
import json
from datetime import datetime


# ====================================================================================
# 🎯 ЦЕНТРАЛЬНАЯ КОНФИГУРАЦИЯ - ВСЕ ГИПЕРПАРАМЕТРЫ В ОДНОМ МЕСТЕ
# ====================================================================================

class Config:
    """
    🎯 ВСЕ ГИПЕРПАРАМЕТРЫ ЗДЕСЬ!
    Измените любой параметр - остальные останутся по умолчанию

    📈 ДЛЯ УЛУЧШЕНИЯ КАЧЕСТВА (медленнее, но лучше):
    - NUM_ITERATIONS = 30000-50000 (больше итераций)
    - CHANNELS = 128 (больше каналов)
    - NUM_BLOCKS = 12-16 (больше блоков)
    - CROP_SIZE = 96-128 (больше размер патчей)
    - INITIAL_LR = 0.00005 (меньше learning rate)
    - LOSS_EDGE_WEIGHT = 0.3 (больше вес на края)
    - LOSS_HF_WEIGHT = 0.2 (больше вес на детали)

    🚀 ДЛЯ x8 УВЕЛИЧЕНИЯ:
    - SR_FACTOR = 8
    - NUM_ITERATIONS = 30000-40000
    - CROP_SIZE = 48-64
    - NUM_BLOCKS = 12-16 (больше блоков для сложной задачи)
    """

    # === ПУТИ К ФАЙЛАМ ===
    INPUT_FILE = "data/single_amsr2_image_2.npz"  # 👈 ИЗМЕНИТЕ НА СВОЙ ПУТЬ!
    OUTPUT_DIR = "results"  # Папка для сохранения результатов

    # === ОСНОВНЫЕ ПАРАМЕТРЫ ===
    SR_FACTOR = 8  # Фактор увеличения: 4, 8 или 16
    NUM_ITERATIONS = 40000  # Количество итераций (None = авто-выбор по SR_FACTOR)
    CROP_SIZE = None  # Размер патчей (None = авто-выбор по SR_FACTOR)

    # === АРХИТЕКТУРА СЕТИ ===
    CHANNELS = 128  # Количество каналов в скрытых слоях
    NUM_BLOCKS = 16  # Количество Residual Attention блоков
    ATTENTION_REDUCTION = 8  # Степень сжатия в Channel Attention
    USE_SPATIAL_ATTENTION = True  # Использовать Spatial Attention

    # === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
    INITIAL_LR = 0.00005  # Начальный learning rate
    WEIGHT_DECAY = 1e-4  # L2 регуляризация
    ADAM_BETAS = (0.9, 0.999)  # Параметры оптимизатора Adam
    GRADIENT_CLIP = 1.0  # Максимальный градиент

    # === ФУНКЦИЯ ПОТЕРЬ ===
    LOSS_L1_WEIGHT = 0.6  # Вес L1 loss (основная точность)
    LOSS_EDGE_WEIGHT = 0.15  # Вес Edge loss (сохранение краев)
    LOSS_HF_WEIGHT = 0.25  # Вес High-frequency loss (детали)
    NORMALIZE_LOSS = True  # Нормализовать loss для лучшего мониторинга

    # === АУГМЕНТАЦИЯ ДАННЫХ ===
    BLUR_MIN = 0.5  # Минимальная сигма размытия для деградации
    BLUR_MAX = 2.0  # Максимальная сигма размытия
    NOISE_PROBABILITY = 0.8  # Вероятность добавления шума
    NOISE_MIN = 0.001  # Минимальный уровень шума
    NOISE_MAX = 0.01  # Максимальный уровень шума
    BRIGHTNESS_RANGE = (0.9, 1.1)  # Диапазон изменения яркости
    CONTRAST_RANGE = (0.9, 1.1)  # Диапазон изменения контраста

    # === ВАЛИДАЦИЯ И EARLY STOPPING ===
    VALIDATION_FREQ = 1000  # Частота валидации
    PATIENCE = 10  # Терпение для early stopping
    IMPROVEMENT_THRESHOLD = 0.998  # Порог улучшения (99.5%)
    MAX_LOSS_THRESHOLD = 10.0  # Максимальный допустимый loss

    # === INFERENCE ===
    USE_TTA = True  # Test Time Augmentation
    SHARPENING_STRENGTH = 0.5  # Сила постобработки резкости

    # === ДОПОЛНИТЕЛЬНЫЕ ПАРАМЕТРЫ ===
    DEVICE = "auto"  # "auto", "cuda", "cpu" или номер GPU (например, "cuda:1")
    MIXED_PRECISION = True  # Использовать mixed precision (FP16) - только для CUDA
    NUM_WORKERS = 0  # Количество потоков для загрузки данных
    SAVE_INTERMEDIATE = True  # Сохранять промежуточные результаты
    VERBOSE = True  # Подробный вывод

    @classmethod
    def get_auto_params(cls):
        """Автоматический выбор параметров на основе SR_FACTOR"""
        if cls.NUM_ITERATIONS is None:
            cls.NUM_ITERATIONS = {
                4: 15000,
                8: 20000,
                16: 25000
            }.get(cls.SR_FACTOR, 15000)

        if cls.CROP_SIZE is None:
            cls.CROP_SIZE = {
                4: 64,
                8: 48,
                16: 32
            }.get(cls.SR_FACTOR, 64)

    @classmethod
    def to_dict(cls):
        """Конвертирует конфигурацию в словарь"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    @classmethod
    def save(cls, path):
        """Сохраняет конфигурацию в JSON"""
        with open(path, 'w') as f:
            json.dump(cls.to_dict(), f, indent=4)

    @classmethod
    def get_filename_suffix(cls):
        """Создает суффикс для имени файла на основе измененных параметров"""
        # Дефолтные значения
        defaults = {
            'SR_FACTOR': 4,
            'CHANNELS': 64,
            'NUM_BLOCKS': 8,
            'INITIAL_LR': 0.0001,
            'LOSS_L1_WEIGHT': 0.7,
            'LOSS_EDGE_WEIGHT': 0.2,
            'LOSS_HF_WEIGHT': 0.1
        }

        # Находим измененные параметры
        suffix_parts = []
        current_config = cls.to_dict()

        for key, default_value in defaults.items():
            if key in current_config and current_config[key] != default_value:
                # Форматируем значение для имени файла
                value = current_config[key]
                if isinstance(value, float):
                    value_str = f"{value:.0e}" if value < 0.01 else f"{value:.3f}"
                else:
                    value_str = str(value)

                # Сокращаем имя параметра
                short_name = {
                    'SR_FACTOR': 'sr',
                    'CHANNELS': 'ch',
                    'NUM_BLOCKS': 'blocks',
                    'INITIAL_LR': 'lr',
                    'LOSS_L1_WEIGHT': 'l1w',
                    'LOSS_EDGE_WEIGHT': 'edgew',
                    'LOSS_HF_WEIGHT': 'hfw'
                }.get(key, key.lower())

                suffix_parts.append(f"{short_name}{value_str}")

        # Добавляем timestamp для уникальности
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts.append(timestamp)

        return "_".join(suffix_parts)


# ====================================================================================
# ПРИМЕНЯЕМ КОНФИГУРАЦИЮ
# ====================================================================================

# Автоматически выбираем параметры
Config.get_auto_params()


# Настройка устройства
def get_device():
    if Config.DEVICE == "auto":
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif Config.DEVICE.startswith("cuda") and torch.cuda.is_available():
        return torch.device(Config.DEVICE)
    else:
        return torch.device('cpu')


# ====================================================================================
# ОСТАЛЬНОЙ КОД С ИСПОЛЬЗОВАНИЕМ Config
# ====================================================================================

# ==================== CHANNEL ATTENTION MODULE ====================
class ChannelAttention(nn.Module):
    """Легковесный механизм внимания по каналам для усиления важных признаков"""

    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        reduction = Config.ATTENTION_REDUCTION
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling path
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Max pooling path
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


# ==================== SPATIAL ATTENTION MODULE ====================
class SpatialAttention(nn.Module):
    """Пространственное внимание для фокусировки на важных областях"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


# ==================== RESIDUAL ATTENTION BLOCK ====================
class ResidualAttentionBlock(nn.Module):
    """Остаточный блок с механизмом внимания"""

    def __init__(self, channels, kernel_size=3, use_spatial=True):
        super(ResidualAttentionBlock, self).__init__()

        # Convolutions
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=True)

        # GroupNorm instead of BatchNorm (works better with batch_size=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)

        # Attention modules
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention() if use_spatial and Config.USE_SPATIAL_ATTENTION else None

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)

        # Apply attention
        out = self.channel_attention(out)
        if self.spatial_attention is not None:
            out = self.spatial_attention(out)

        # Residual connection
        out = out + residual
        out = self.relu(out)

        return out


# ==================== ENHANCED ZSSR NETWORK WITH ATTENTION ====================
class AttentionZSSRNet(nn.Module):
    def __init__(self, input_channels=1):
        super(AttentionZSSRNet, self).__init__()

        channels = Config.CHANNELS
        num_blocks = Config.NUM_BLOCKS
        sr_factor = Config.SR_FACTOR

        self.sr_factor = sr_factor

        # Initial feature extraction with larger kernel for better context
        self.conv_first = nn.Conv2d(input_channels, channels, kernel_size=7, padding=3, bias=True)

        # Residual attention blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Use spatial attention every 2 blocks to reduce computation
            use_spatial = (i % 2 == 0)
            self.res_blocks.append(ResidualAttentionBlock(channels, use_spatial=use_spatial))

        # Feature fusion before upsampling
        self.fusion = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        # Sub-pixel convolution for upsampling (more efficient than deconv)
        self.upscale_layers = nn.ModuleList()
        num_upscale = int(np.log2(sr_factor))
        for _ in range(num_upscale):
            self.upscale_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True)
                )
            )

        # Output projection
        self.conv_last = nn.Conv2d(channels, input_channels, kernel_size=3, padding=1, bias=True)

        # Skip connection for residual learning
        self.skip_upsample = nn.Upsample(scale_factor=sr_factor, mode='bicubic', align_corners=False)

    def forward(self, x):
        # Bicubic upsampling for skip connection
        bicubic = self.skip_upsample(x)

        # Feature extraction
        feat = self.conv_first(x)
        feat = F.relu(feat, inplace=True)

        # Pass through residual attention blocks
        for block in self.res_blocks:
            feat = block(feat)

        # Feature fusion
        feat = self.fusion(feat)

        # Progressive upsampling
        for upscale in self.upscale_layers:
            feat = upscale(feat)

        # Final projection
        out = self.conv_last(feat)

        # Residual learning - predict the difference
        return out + bicubic


# ==================== EDGE-AWARE GRADIENT LOSS ====================
class EdgeAwareGradientLoss(nn.Module):
    """Потеря на основе градиентов с акцентом на края для улучшения резкости"""

    def __init__(self):
        super(EdgeAwareGradientLoss, self).__init__()

        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, output, target):
        # FIXED: Move sobel filters to the same device and dtype as input
        sobel_x = self.sobel_x.to(device=output.device, dtype=output.dtype)
        sobel_y = self.sobel_y.to(device=output.device, dtype=output.dtype)

        # Apply Sobel filters
        output_grad_x = F.conv2d(output, sobel_x, padding=1)
        output_grad_y = F.conv2d(output, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        # Compute gradient magnitude
        output_grad_mag = torch.sqrt(output_grad_x ** 2 + output_grad_y ** 2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)

        # L1 loss on gradient magnitudes
        return F.l1_loss(output_grad_mag, target_grad_mag)


# ==================== SHARPNESS-ENHANCED LOSS ====================
class SharpnessEnhancedLoss(nn.Module):
    """Комбинированная функция потерь с акцентом на резкость"""

    def __init__(self):
        super(SharpnessEnhancedLoss, self).__init__()
        self.alpha = Config.LOSS_L1_WEIGHT
        self.beta = Config.LOSS_EDGE_WEIGHT
        self.gamma = Config.LOSS_HF_WEIGHT

        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeAwareGradientLoss()

    def high_frequency_loss(self, output, target):
        """Потеря на высоких частотах для сохранения деталей"""
        # Simple high-pass filter
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                              dtype=torch.float32).view(1, 1, 3, 3)

        # FIXED: Move kernel to the same device and dtype as input
        kernel = kernel.to(device=output.device, dtype=output.dtype)

        output_hf = F.conv2d(output, kernel, padding=1)
        target_hf = F.conv2d(target, kernel, padding=1)

        return F.l1_loss(output_hf, target_hf)

    def forward(self, output, target):
        l1 = self.l1_loss(output, target)
        edge = self.edge_loss(output, target)
        hf = self.high_frequency_loss(output, target)

        # Normalize losses if configured
        if Config.NORMALIZE_LOSS:
            # Normalize each component to similar scale
            l1_norm = l1
            edge_norm = edge * 0.5  # Edge loss tends to be ~2x larger
            hf_norm = hf * 0.3  # HF loss tends to be ~3x larger

            total_loss = self.alpha * l1_norm + self.beta * edge_norm + self.gamma * hf_norm
        else:
            total_loss = self.alpha * l1 + self.beta * edge + self.gamma * hf

        return total_loss, {'l1': l1.item(), 'edge': edge.item(), 'hf': hf.item()}


# ==================== DIRECT ARRAY DATA SAMPLER ====================
class DirectArrayDataSampler:
    """Работает напрямую с numpy массивами без преобразования в изображения"""

    def __init__(self, data_array):
        self.data = data_array.astype(np.float32)
        self.sr_factor = Config.SR_FACTOR
        self.crop_size = Config.CROP_SIZE

        # Pre-compute multi-scale versions
        self.scales = self._create_multiscale_data()

    def _create_multiscale_data(self):
        """Создает версии данных в разных масштабах"""
        h, w = self.data.shape
        scales = []

        # Create different scales
        scale_factors = np.linspace(0.5, 1.0, 8)

        for scale in scale_factors:
            if scale == 1.0:
                scales.append(self.data)
            else:
                new_h, new_w = int(h * scale), int(w * scale)
                # Anti-aliasing before downsampling
                sigma = 1.0 / scale
                blurred = cv2.GaussianBlur(self.data, (0, 0), sigma)
                scaled = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scales.append(scaled)

        return scales

    def _apply_degradation(self, hr_patch):
        """Применяет реалистичную деградацию для создания LR патча"""
        # Random blur strength
        blur_sigma = np.random.uniform(Config.BLUR_MIN, Config.BLUR_MAX)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(hr_patch, (0, 0), blur_sigma)

        # Downscale
        h, w = hr_patch.shape
        lr_h, lr_w = h // self.sr_factor, w // self.sr_factor
        lr = cv2.resize(blurred, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)

        # Add noise
        if np.random.random() < Config.NOISE_PROBABILITY:
            noise_sigma = np.random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
            noise = np.random.normal(0, noise_sigma, lr.shape).astype(np.float32)
            lr = lr + noise

        # Upscale back
        lr_upscaled = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

        return np.clip(lr_upscaled, 0, 1)

    def get_training_pair(self):
        """Получает пару HR-LR для обучения"""
        # Random scale selection
        scale_idx = np.random.randint(len(self.scales))
        data = self.scales[scale_idx]

        h, w = data.shape

        # ВАЖНО: для обучения используем HR патчи размером crop_size * sr_factor
        # чтобы после даунсэмплинга получить LR патчи размером crop_size
        hr_crop_size = self.crop_size * self.sr_factor

        # Ensure we can crop
        if h < hr_crop_size or w < hr_crop_size:
            # Pad if necessary
            pad_h = max(0, hr_crop_size - h)
            pad_w = max(0, hr_crop_size - w)
            data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = data.shape

        # Random crop with bias towards high-variance regions
        if np.random.random() < 0.7:  # 70% chance to select informative region
            # Compute local variance
            var_map = cv2.Laplacian(data, cv2.CV_32F)
            var_map = cv2.GaussianBlur(np.abs(var_map), (11, 11), 0)

            # Find high variance region
            valid_h = h - hr_crop_size
            valid_w = w - hr_crop_size

            if valid_h > 0 and valid_w > 0:
                # Sample from top 20% variance regions
                var_thresh = np.percentile(var_map[:valid_h, :valid_w], 80)
                high_var_mask = var_map[:valid_h, :valid_w] > var_thresh
                high_var_coords = np.argwhere(high_var_mask)

                if len(high_var_coords) > 0:
                    idx = np.random.randint(len(high_var_coords))
                    y, x = high_var_coords[idx]
                else:
                    y = np.random.randint(0, valid_h + 1)
                    x = np.random.randint(0, valid_w + 1)
            else:
                y, x = 0, 0
        else:
            # Random crop
            y = np.random.randint(0, h - hr_crop_size + 1)
            x = np.random.randint(0, w - hr_crop_size + 1)

        # Extract HR patch (размер: hr_crop_size × hr_crop_size)
        hr_patch = data[y:y + hr_crop_size, x:x + hr_crop_size].copy()

        # Random augmentations
        if np.random.random() < 0.5:
            hr_patch = np.fliplr(hr_patch).copy()
        if np.random.random() < 0.5:
            hr_patch = np.flipud(hr_patch).copy()
        if np.random.random() < 0.5:
            k = np.random.randint(1, 4)
            hr_patch = np.rot90(hr_patch, k).copy()

        # Brightness/contrast augmentation
        if np.random.random() < 0.3:
            brightness = np.random.uniform(*Config.BRIGHTNESS_RANGE)
            hr_patch = np.clip(hr_patch * brightness, 0, 1)

        if np.random.random() < 0.3:
            contrast = np.random.uniform(*Config.CONTRAST_RANGE)
            mean = np.mean(hr_patch)
            hr_patch = np.clip((hr_patch - mean) * contrast + mean, 0, 1)

        # Create LR patch by downsampling HR patch
        # LR будет размером crop_size × crop_size
        lr_patch = cv2.resize(hr_patch, (self.crop_size, self.crop_size),
                              interpolation=cv2.INTER_LINEAR)

        # Apply degradation to LR
        lr_patch = self._apply_degradation_to_lr(lr_patch)

        # Convert to tensors
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0).unsqueeze(0).float()
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0).float()

        return hr_tensor, lr_tensor

    def _apply_degradation_to_lr(self, lr_patch):
        """Применяет деградацию к LR патчу"""
        # Random blur
        if np.random.random() < 0.7:
            blur_sigma = np.random.uniform(Config.BLUR_MIN, Config.BLUR_MAX)
            lr_patch = cv2.GaussianBlur(lr_patch, (0, 0), blur_sigma)

        # Add noise
        if np.random.random() < Config.NOISE_PROBABILITY:
            noise_sigma = np.random.uniform(Config.NOISE_MIN, Config.NOISE_MAX)
            noise = np.random.normal(0, noise_sigma, lr_patch.shape).astype(np.float32)
            lr_patch = lr_patch + noise

        return np.clip(lr_patch, 0, 1)


# ==================== TRAINING WITH LOSS MONITORING ====================
def train_with_attention(model, data_array):
    """Обучение с мониторингом потерь и early stopping"""

    device = get_device()
    model = model.to(device)

    if Config.VERBOSE:
        print(f"\n🎯 Training AttentionZSSR Model")
        print(f"Device: {device}")
        print(f"SR Factor: {Config.SR_FACTOR}x")
        print(f"Initial LR: {Config.INITIAL_LR}")
        print(f"Iterations: {Config.NUM_ITERATIONS}")

    # Loss function
    criterion = SharpnessEnhancedLoss()

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=Config.INITIAL_LR,
                            betas=Config.ADAM_BETAS, weight_decay=Config.WEIGHT_DECAY)

    # Scheduler with minimum LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_ITERATIONS, eta_min=1e-6, last_epoch=-1)

    # Mixed precision training for efficiency
    if Config.MIXED_PRECISION and device.type == 'cuda':
        try:
            # Try new PyTorch syntax first
            from torch.amp import GradScaler as NewGradScaler
            scaler = NewGradScaler('cuda')
            use_amp = True
            amp_style = 'new'
        except (ImportError, TypeError):
            # Fall back to old syntax
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            use_amp = True
            amp_style = 'old'
    else:
        scaler = None
        use_amp = False
        amp_style = None

    # Data sampler
    sampler = DirectArrayDataSampler(data_array)

    # Loss tracking
    loss_history = deque(maxlen=100)
    loss_components = {'l1': deque(maxlen=100), 'edge': deque(maxlen=100), 'hf': deque(maxlen=100)}
    best_loss = float('inf')
    patience_counter = 0

    # Loss explosion protection
    loss_explosion_counter = 0

    with tqdm.tqdm(total=Config.NUM_ITERATIONS, disable=not Config.VERBOSE) as pbar:
        for batch_idx in range(Config.NUM_ITERATIONS):
            model.train()

            # Get training pair
            hr, lr = sampler.get_training_pair()
            hr, lr = hr.to(device), lr.to(device)

            # Forward pass
            if use_amp:
                if amp_style == 'new':
                    # New PyTorch syntax
                    from torch.amp import autocast
                    with autocast('cuda', dtype=torch.float16):
                        output = model(lr)
                        loss, components = criterion(output, hr)
                else:
                    # Old PyTorch syntax
                    from torch.cuda.amp import autocast
                    with autocast():
                        output = model(lr)
                        loss, components = criterion(output, hr)
            else:
                output = model(lr)
                loss, components = criterion(output, hr)

            # Check for loss explosion
            if loss.item() > Config.MAX_LOSS_THRESHOLD:
                loss_explosion_counter += 1
                if loss_explosion_counter > 3:
                    print(f"\n⚠️ Loss explosion detected! Stopping training.")
                    break
            else:
                loss_explosion_counter = 0

            # Backward pass
            optimizer.zero_grad()

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRADIENT_CLIP)
                optimizer.step()

            scheduler.step()

            # Track losses
            loss_val = loss.item()
            loss_history.append(loss_val)
            for key, value in components.items():
                loss_components[key].append(value)

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(
                f"Loss: {loss_val:.4f} | "
                f"L1: {components['l1']:.4f} | "
                f"Edge: {components['edge']:.4f} | "
                f"HF: {components['hf']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            pbar.update()

            # Validation and early stopping
            if batch_idx > 0 and batch_idx % Config.VALIDATION_FREQ == 0:
                avg_loss = np.mean(loss_history)

                if avg_loss < best_loss * Config.IMPROVEMENT_THRESHOLD:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()

                    # Save intermediate checkpoint
                    if Config.SAVE_INTERMEDIATE:
                        checkpoint_path = os.path.join(Config.OUTPUT_DIR, f'checkpoint_iter{batch_idx}.pth')
                        torch.save({
                            'model_state_dict': best_model_state,
                            'iteration': batch_idx,
                            'loss': best_loss
                        }, checkpoint_path)
                else:
                    patience_counter += 1

                if patience_counter >= Config.PATIENCE:
                    print(f"\n✋ Early stopping at iteration {batch_idx}")
                    model.load_state_dict(best_model_state)
                    break

    return model


# ==================== INFERENCE WITH TTA ====================
def inference_with_tta(model, data_array):
    """Inference с Test Time Augmentation для лучшего качества"""
    device = get_device()
    model = model.to(device).eval()

    # Prepare input
    h, w = data_array.shape
    data_tensor = torch.from_numpy(data_array).float().unsqueeze(0).unsqueeze(0)

    if not Config.USE_TTA:
        # Без TTA
        with torch.no_grad():
            input_tensor = data_tensor.to(device)
            output = model(input_tensor)
            result = output.cpu().squeeze().numpy()
    else:
        # С TTA
        augmented_outputs = []

        with torch.no_grad():
            # 1. Original
            input_tensor = data_tensor.to(device)
            output = model(input_tensor)
            augmented_outputs.append(output.cpu())

            # 2. Horizontal flip
            input_flipped = torch.flip(input_tensor, dims=[3])
            output_flipped = model(input_flipped)
            output_flipped = torch.flip(output_flipped, dims=[3])
            augmented_outputs.append(output_flipped.cpu())

            # 3. Vertical flip
            input_flipped = torch.flip(input_tensor, dims=[2])
            output_flipped = model(input_flipped)
            output_flipped = torch.flip(output_flipped, dims=[2])
            augmented_outputs.append(output_flipped.cpu())

            # 4. 90 degree rotations
            for k in [1, 2, 3]:
                input_rotated = torch.rot90(input_tensor, k, dims=[2, 3])
                output_rotated = model(input_rotated)
                output_rotated = torch.rot90(output_rotated, -k, dims=[2, 3])
                augmented_outputs.append(output_rotated.cpu())

        # Average all predictions
        final_output = torch.stack(augmented_outputs).mean(dim=0)
        result = final_output.squeeze().numpy()

    # Post-processing for sharpness
    result = enhance_sharpness(result)

    return np.clip(result, 0, 1)


def enhance_sharpness(image):
    """Пост-обработка для улучшения резкости"""
    # Unsharp masking
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    sharpened = image + Config.SHARPENING_STRENGTH * (image - blurred)

    # Ensure float32
    sharpened = sharpened.astype(np.float32)

    # Adaptive sharpening based on local variance
    local_var = cv2.Laplacian(image, cv2.CV_32F)
    local_var = cv2.GaussianBlur(np.abs(local_var), (5, 5), 0)

    # Normalize variance map
    var_min, var_max = local_var.min(), local_var.max()
    if var_max > var_min:
        var_norm = (local_var - var_min) / (var_max - var_min)
    else:
        var_norm = np.zeros_like(local_var)

    # Apply adaptive sharpening
    result = image * (1 - 0.3 * var_norm) + sharpened * (0.3 * var_norm)

    return np.clip(result, 0, 1)


# ==================== MAIN PROCESSING FUNCTION ====================
def process_satellite_data(npz_path):
    """Основная функция обработки спутниковых данных"""

    print(f"\n🚀 Enhanced ZSSR v3 with Easy Configuration")
    print(f"📡 Processing: {npz_path}")
    print(f"🔍 Target SR: {Config.SR_FACTOR}x")

    # Load data
    with np.load(npz_path, allow_pickle=True) as data:
        temperature = data['temperature'].astype(np.float32)
        metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']

    # Extract scale factor
    if isinstance(metadata, dict):
        scale_factor = metadata.get('scale_factor', 0.01)
    else:
        import re
        metadata_str = str(metadata)
        scale_match = re.search(r"'scale_factor': ([0-9.e-]+)", metadata_str)
        scale_factor = float(scale_match.group(1)) if scale_match else 0.01

    # Process temperature data
    temp_data = temperature * scale_factor

    # Handle missing values
    valid_mask = temp_data != 0
    if np.sum(valid_mask) > 0:
        median_temp = np.median(temp_data[valid_mask])
        temp_data[~valid_mask] = median_temp

    # Normalize with percentile clipping
    p_low, p_high = np.percentile(temp_data[valid_mask], [1, 99])
    temp_data = np.clip(temp_data, p_low, p_high)
    temp_min, temp_max = p_low, p_high
    normalized_data = (temp_data - temp_min) / (temp_max - temp_min)

    print(f"📊 Data shape: {normalized_data.shape}")
    print(f"🌡️ Temperature range: {temp_min:.1f} - {temp_max:.1f} K")

    # Initialize model
    model = AttentionZSSRNet(input_channels=1)

    # Custom initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    # Train model
    model = train_with_attention(model, normalized_data)

    # Apply model with TTA
    print("\n🖼️ Generating super-resolved output...")
    enhanced_normalized = inference_with_tta(model, normalized_data)

    # Denormalize
    enhanced_temp = enhanced_normalized * (temp_max - temp_min) + temp_min

    # Save results with temperature range info
    print(f"\n📊 Temperature statistics:")
    print(f"   Original: min={original_temp.min():.2f} K, max={original_temp.max():.2f} K")
    print(f"   Enhanced: min={enhanced_temp.min():.2f} K, max={enhanced_temp.max():.2f} K")

    save_results_with_comparison(normalized_data, enhanced_normalized, temp_min, temp_max)

    # Generate filename with config suffix
    suffix = Config.get_filename_suffix()

    # Save model
    model_path = os.path.join(Config.OUTPUT_DIR, f'attention_zssr_{suffix}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': Config.to_dict(),
        'temperature_range': (temp_min, temp_max)
    }, model_path)

    # Save config (without classmethods)
    config_path = os.path.join(Config.OUTPUT_DIR, f'config_{suffix}.json')
    config_dict = {
        k: v for k, v in Config.__dict__.items()
        if not k.startswith('_') and not callable(getattr(Config, k))
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(f"\n💾 Model saved: {model_path}")
    print(f"📄 Config saved: {config_path}")

    return enhanced_temp, model


def save_results_with_comparison(original_norm, enhanced_norm, temp_min, temp_max):
    """Сохранение результатов с визуальным сравнением"""

    # Denormalize for visualization
    original_temp = original_norm * (temp_max - temp_min) + temp_min
    enhanced_temp = enhanced_norm * (temp_max - temp_min) + temp_min

    print(f"\n📸 Saving results...")

    # Generate filename suffix
    suffix = Config.get_filename_suffix()

    # Calculate proper figure dimensions based on aspect ratio
    h, w = original_temp.shape
    ratio = w / h
    base_height = 12  # inches
    base_width = ratio * base_height

    # Create comparison figure with proper aspect ratio
    fig = plt.figure(figsize=(base_width * 2.2, base_height))

    # Original
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(original_temp, cmap='turbo', aspect='auto')
    ax1.set_title(f'Original ({original_temp.shape[0]}×{original_temp.shape[1]})\n~10 km/pixel', fontsize=16)
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.set_ylabel('Temperature (K)', fontsize=12)

    # Enhanced
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(enhanced_temp, cmap='turbo', aspect='auto')
    ax2.set_title(
        f'Enhanced ({enhanced_temp.shape[0]}×{enhanced_temp.shape[1]})\n~{10 / Config.SR_FACTOR:.1f} km/pixel',
        fontsize=16)
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.set_ylabel('Temperature (K)', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'comparison_{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save individual images with proper aspect ratio
    # Original
    plt.figure(figsize=(base_width, base_height))
    plt.imshow(original_temp, cmap='turbo', aspect='auto')
    plt.axis('off')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'original_{suffix}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Enhanced
    enhanced_h, enhanced_w = enhanced_temp.shape
    enhanced_ratio = enhanced_w / enhanced_h
    enhanced_width = enhanced_ratio * base_height

    plt.figure(figsize=(enhanced_width, base_height))
    plt.imshow(enhanced_temp, cmap='turbo', aspect='auto')
    plt.axis('off')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'enhanced_{suffix}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Detail comparison - zoom on interesting region
    h, w = original_temp.shape
    crop_size = min(100, h // 4)

    # Find region with high variance (most interesting)
    var_map = cv2.Laplacian(original_norm.astype(np.float32), cv2.CV_32F)
    var_map = cv2.GaussianBlur(np.abs(var_map), (21, 21), 0)

    # Get coordinates of maximum variance
    max_var_idx = np.unravel_index(
        np.argmax(var_map[crop_size:-crop_size, crop_size:-crop_size]),
        var_map[crop_size:-crop_size, crop_size:-crop_size].shape
    )
    center_h = max_var_idx[0] + crop_size
    center_w = max_var_idx[1] + crop_size

    # Create detail comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original crop
    crop_orig = original_temp[center_h - crop_size // 2:center_h + crop_size // 2,
                center_w - crop_size // 2:center_w + crop_size // 2]
    axes[0].imshow(crop_orig, cmap='turbo', aspect='auto')
    axes[0].set_title('Original Detail', fontsize=14)
    axes[0].axis('off')

    # Bicubic upsampled crop (for comparison)
    crop_bicubic = cv2.resize(crop_orig, (crop_size * Config.SR_FACTOR, crop_size * Config.SR_FACTOR),
                              interpolation=cv2.INTER_CUBIC)
    axes[1].imshow(crop_bicubic, cmap='turbo', aspect='auto')
    axes[1].set_title('Bicubic Interpolation', fontsize=14)
    axes[1].axis('off')

    # Enhanced crop
    sr = Config.SR_FACTOR
    crop_enh = enhanced_temp[center_h * sr - crop_size * sr // 2:center_h * sr + crop_size * sr // 2,
               center_w * sr - crop_size * sr // 2:center_w * sr + crop_size * sr // 2]
    axes[2].imshow(crop_enh, cmap='turbo', aspect='auto')
    axes[2].set_title('AttentionZSSR (Ours)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'detail_{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save numpy arrays
    np.save(os.path.join(Config.OUTPUT_DIR, f'enhanced_data_{suffix}.npy'), enhanced_temp)
    np.save(os.path.join(Config.OUTPUT_DIR, f'original_data_{suffix}.npy'), original_temp)

    print("✅ Results saved:")
    print(f"   📁 {Config.OUTPUT_DIR}/")
    print(f"      ├── comparison_{suffix}.png")
    print(f"      ├── original_{suffix}.png")
    print(f"      ├── enhanced_{suffix}.png")
    print(f"      ├── detail_{suffix}.png")
    print(f"      ├── enhanced_data_{suffix}.npy")
    print(f"      └── original_data_{suffix}.npy")


# ==================== MAIN EXECUTION ====================
def main():
    """Главная функция"""
    # Создаем выходную директорию
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Проверка входного файла
    if not os.path.exists(Config.INPUT_FILE):
        print(f"❌ Файл не найден: {Config.INPUT_FILE}")
        print("📝 Пожалуйста, измените Config.INPUT_FILE на правильный путь")
        return

    print(f"\n{'=' * 60}")
    print(f"🚀 Enhanced ZSSR v3 - Easy Hyperparameter Configuration")
    print(f"{'=' * 60}")
    print(f"\n📋 Current Configuration:")
    print(f"   SR Factor: {Config.SR_FACTOR}x")
    print(f"   Iterations: {Config.NUM_ITERATIONS}")
    print(f"   Crop Size: {Config.CROP_SIZE}")
    print(f"   Channels: {Config.CHANNELS}")
    print(f"   Blocks: {Config.NUM_BLOCKS}")
    print(f"   Learning Rate: {Config.INITIAL_LR}")
    print(f"   Loss Weights: L1={Config.LOSS_L1_WEIGHT}, Edge={Config.LOSS_EDGE_WEIGHT}, HF={Config.LOSS_HF_WEIGHT}")
    print(f"{'=' * 60}\n")

    try:
        # Process data
        enhanced_temp, model = process_satellite_data(Config.INPUT_FILE)

        print(f"\n✅ Processing complete!")
        print(f"📁 All results saved in: {Config.OUTPUT_DIR}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()