import cv2
import matplotlib.pyplot as plt

# 1. 이미지 로드 (사용자의 파일 경로에 맞춰 수정하세요)
image_path = './path-to-gpx/data/images/083.jpg'  # 마라톤 코스 이미지 파일명
image = cv2.imread(image_path)

# 이미지 로드 확인
if image is None:
    print(f"이미지를 찾을 수 없습니다: {image_path}")
else:
    # 2. 확인하고 싶은 픽셀 좌표 설정
    img_x, img_y = 612.0, 623.0
    
    # 3. 빨간색 점 그리기 (OpenCV는 BGR 순서이므로 빨간색은 (0, 0, 255))
    # 좌표는 정수형(int)으로 변환하여 입력합니다.
    center_coordinates = (int(img_x), int(img_y))
    radius = 10         # 점의 크기
    color = (0, 0, 255) # 빨간색
    thickness = -1      # -1은 내부를 채운 원을 의미합니다.
    
    cv2.circle(image, center_coordinates, radius, color, thickness)

    # 4. 결과 출력
    # OpenCV 이미지는 BGR이므로 Matplotlib 출력을 위해 RGB로 변환합니다.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(f"Point at ({img_x}, {img_y})")
    plt.axis('on') # 픽셀 좌표 확인을 위해 축 표시
    plt.show()