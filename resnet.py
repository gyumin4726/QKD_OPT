import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

class BasicBlock(nn.Module):
    """
    ResNet의 기본 블록 (Basic Block)
    - 2개의 3x3 컨볼루션 레이어로 구성
    - skip connection을 통해 입력을 출력에 직접 더함 (residual connection)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection을 위한 1x1 컨볼루션 (입력과 출력 채널이 다를 때)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # 메인 경로 (두 개의 컨볼루션 레이어)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Skip connection (입력을 출력에 직접 더함)
        out += self.shortcut(x)
        
        # 최종 활성화 함수
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    간단한 ResNet 모델
    - CIFAR-10 데이터셋에 최적화된 구조
    - 여러 개의 BasicBlock을 쌓아서 깊은 네트워크 구성
    """
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        # 초기 컨볼루션 레이어 (입력 이미지를 64개 채널로 변환)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet 레이어들 (각 레이어는 여러 개의 BasicBlock으로 구성)
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)
        
        # 최종 분류 레이어
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        ResNet 레이어를 만드는 헬퍼 함수
        - 첫 번째 블록에서만 stride를 적용하여 크기 축소
        - 나머지 블록들은 stride=1로 유지
        """
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 초기 컨볼루션
        x = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet 레이어들
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 글로벌 평균 풀링 (공간 차원을 1x1로 축소)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 1차원으로 펼치기
        
        # 최종 분류
        x = self.fc(x)
        return x


def create_resnet18():
    """
    ResNet-18 모델 생성 (각 레이어에 2개씩 블록)
    """
    return ResNet([2, 2, 2, 2])


def create_resnet34():
    """
    ResNet-34 모델 생성 (각 레이어에 3, 4, 6, 3개 블록)
    """
    return ResNet([3, 4, 6, 3])


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """
    ResNet 모델을 학습시키는 함수
    
    Args:
        model: 학습할 ResNet 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        epochs: 학습 에포크 수
        learning_rate: 학습률
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 학습 기록을 위한 리스트
    train_losses = []
    val_accuracies = []
    
    print(f"학습 시작 - 디바이스: {device}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            output = model(data)
            loss = criterion(output, target)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 진행 상황 출력
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # 학습률 스케줄링
        scheduler.step()
        
        # 검증
        val_accuracy = evaluate_model(model, val_loader, device)
        
        # 기록 저장
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} 완료 - '
              f'평균 학습 손실: {avg_train_loss:.4f}, '
              f'검증 정확도: {val_accuracy:.2f}%')
        print('-' * 50)
    
    return train_losses, val_accuracies


def evaluate_model(model, data_loader, device):
    """
    모델의 성능을 평가하는 함수
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_training_history(train_losses, val_accuracies):
    """
    학습 과정을 시각화하는 함수
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 학습 손실 그래프
    ax1.plot(train_losses)
    ax1.set_title('학습 손실 (Training Loss)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 검증 정확도 그래프
    ax2.plot(val_accuracies)
    ax2.set_title('검증 정확도 (Validation Accuracy)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# 사용 예시
if __name__ == "__main__":
    print("=== ResNet 구현 및 학습 예시 ===")
    print()
    
    # 1. 모델 생성
    print("1. ResNet-18 모델 생성 중...")
    model = create_resnet18()
    print(f"모델 구조:\n{model}")
    print()
    
    # 2. 더미 데이터 생성 (실제 사용시에는 CIFAR-10 데이터셋 사용)
    print("2. 더미 데이터 생성 중...")
    batch_size = 32
    num_samples = 1000
    
    # 랜덤 이미지 데이터 (3채널, 32x32 픽셀)
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, 10, (num_samples,))
    X_val = torch.randn(200, 3, 32, 32)
    y_val = torch.randint(0, 10, (200,))
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    print()
    
    # 3. 모델 학습
    print("3. 모델 학습 시작...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        epochs=5, learning_rate=0.001
    )
    
    # 4. 결과 시각화
    print("4. 학습 결과 시각화...")
    plot_training_history(train_losses, val_accuracies)
    
    print("학습 완료!")
