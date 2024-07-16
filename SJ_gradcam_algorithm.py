# %%
import numpy as np

def SJ_activate(heatmap):
    # heatmap에서 0을 제외한 값의 위치를 찾기
    non_zero_positions = np.where(heatmap != 0)

    # 가장 위, 가장 아래, 가장 왼쪽, 가장 오른쪽에 해당하는 좌표를 구하기
    top = np.min(non_zero_positions[0])
    bottom = np.max(non_zero_positions[0])
    left = np.min(non_zero_positions[1])
    right = np.max(non_zero_positions[1])

    # 길이 구하기 (행의 길이와 열의 길이)
    height = bottom - top + 1
    width = right - left + 1

    rows, cols = np.where(heatmap != 0)

    #각 좌표 구하기
    top = np.min(rows)
    bottom = np.max(rows)
    left = np.min(cols)
    right = np.max(cols)
    top_coord = (top, cols[np.argmin(rows)])
    bottom_coord = (bottom, cols[np.argmax(rows)])
    left_coord = (rows[np.argmin(cols)], left)
    right_coord = (rows[np.argmax(cols)], right)

    #가로, 세로 길이 구하기
    hor_length = right_coord[1] - left_coord[1] #가로
    ver_length = bottom_coord[0] - top_coord[0] #세로

    #가로 좌표 지정
    hor_0 = left_coord[1]
    hor_1_3 = int(left_coord[1] + 0.33*hor_length)
    hor_1_2 = int(left_coord[1] + 0.5*hor_length)
    hor_2_3 = int(left_coord[1] + 0.66*hor_length)
    hor_1 = right_coord[1]
    #세로 좌표 지정
    ver_0 = top_coord[0]
    ver_1_3 = int(top_coord[0] + 0.33*ver_length)
    ver_1_2 = int(top_coord[0] + 0.5*ver_length)
    ver_2_3 = int(top_coord[0] + 0.66*ver_length)
    ver_1 = bottom_coord[0]
    #중앙 범위 지정
    area_center = heatmap[ver_1_3:ver_2_3, hor_1_3:hor_2_3]
    #가장자리 범위 지정
    area_edge_1 = heatmap[ver_0:ver_1_2,hor_0:hor_1_2] #왼위
    area_edge_2 = heatmap[ver_1_2:ver_1,hor_0:hor_1_2] #오위
    area_edge_3 = heatmap[ver_0:ver_1_2,hor_1_2:hor_1] #왼아래
    area_edge_4 = heatmap[ver_1_2:ver_1,hor_1_2:hor_1] #오아래

    """
    #범위 시각화 코드
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(heatmap, cmap='gray', alpha=0.5)
    colors = ['PuBu', 'Oranges_r', 'Spectral_r', 'Set2_r','Accent']  # 사용할 색상들
    edges = [(ver_0, ver_1_2, hor_0, hor_1_2), 
            (ver_1_2, ver_1, hor_0, hor_1_2),
            (ver_0, ver_1_2, hor_1_2, hor_1), 
            (ver_1_2, ver_1, hor_1_2, hor_1),
            (ver_1_3,ver_2_3, hor_1_3,hor_2_3)]
    for (color, (h_start, h_end, v_start, v_end)) in zip(colors, edges):
        edge_mask = np.zeros_like(heatmap)
        edge_mask[h_start:h_end, v_start:v_end] = 1
        plt.imshow(edge_mask, cmap=color, alpha=0.3, interpolation='nearest')
    plt.colorbar()
    plt.show()
    """
    #감별 요점 if문
    # 전체 heatmap에 대한 마스크 생성, 이 마스크는 area_center 영역을 제외한 영역에서 True
    total_mask = np.ones_like(heatmap, dtype=bool)  # 전체를 True로 초기화
    total_mask[ver_1_3:ver_2_3, hor_1_3:hor_2_3] = False  # area_center 영역을 False로 설정

    # 각 area_edge 영역에 대한 마스크 생성
    edge_1_mask = np.zeros_like(heatmap, dtype=bool)
    edge_1_mask[ver_0:ver_1_2, hor_0:hor_1_2] = True

    edge_2_mask = np.zeros_like(heatmap, dtype=bool)
    edge_2_mask[ver_1_2:ver_1, hor_0:hor_1_2] = True

    edge_3_mask = np.zeros_like(heatmap, dtype=bool)
    edge_3_mask[ver_0:ver_1_2, hor_1_2:hor_1] = True

    edge_4_mask = np.zeros_like(heatmap, dtype=bool)
    edge_4_mask[ver_1_2:ver_1, hor_1_2:hor_1] = True

    # 겹치는 부분 제외
    area_edge_1_non_overlap = heatmap[edge_1_mask & total_mask]
    area_edge_2_non_overlap = heatmap[edge_2_mask & total_mask]
    area_edge_3_non_overlap = heatmap[edge_3_mask & total_mask]
    area_edge_4_non_overlap = heatmap[edge_4_mask & total_mask]

    # heatmap에서 0이 아닌 픽셀들의 평균 값 계산
    non_zero_pixels = heatmap[heatmap != 0]
    non_zero_mean = np.mean(non_zero_pixels)

    # area_center에서 0이 아닌 픽셀들을 찾음
    area_center_non_zero = area_center[area_center != 0]
    area_edge_1_non_zero = area_edge_1_non_overlap[area_edge_1_non_overlap != 0]
    area_edge_2_non_zero = area_edge_2_non_overlap[area_edge_2_non_overlap != 0]
    area_edge_3_non_zero = area_edge_3_non_overlap[area_edge_3_non_overlap != 0]
    area_edge_4_non_zero = area_edge_4_non_overlap[area_edge_4_non_overlap != 0]


    # area_center 내에서 평균보다 큰 0이 아닌 픽셀들의 비율 계산
    center_mean_ratio = np.sum(area_center_non_zero > non_zero_mean) / len(area_center_non_zero)
    data_max = np.max(area_center_non_zero)
    data_min = np.min(area_center_non_zero)
    std_deviation = np.std(area_center_non_zero)
    normalized_std_deviation = std_deviation / (data_max - data_min)

    area_edge_1_ratio = np.sum(area_edge_1_non_zero > non_zero_mean) / len(area_edge_1_non_zero)
    area_edge_2_ratio = np.sum(area_edge_2_non_zero > non_zero_mean) / len(area_edge_2_non_zero)
    area_edge_3_ratio = np.sum(area_edge_3_non_zero > non_zero_mean) / len(area_edge_3_non_zero)
    area_edge_4_ratio = np.sum(area_edge_4_non_zero > non_zero_mean) / len(area_edge_4_non_zero)

    edge_ratios = [area_edge_1_ratio, area_edge_2_ratio, area_edge_3_ratio, area_edge_4_ratio]
    #edges_above_mean = sum(ratio > 0.8 for ratio in edge_ratios) #edge 구역의 몇개가 0.8 이상인지

    return center_mean_ratio, normalized_std_deviation


#산조인 ZISE
center_line_ZISE = '1. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 자색 또는 자갈색을 띰\n4. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음'
center_noline_ZISE = '1. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n2. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n3. 자색 또는 자갈색을 띰\n4. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음'
only_edge_ZISE = '1. 자색 또는 자갈색을 띰\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n4. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음'

#면조인 ZIMA
center_line_ZIMA = '1. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n2. 회황색을 띰\n3. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n4. 산조인에 비해 두께가 얇고 납작함'
center_noline_ZIMA = '1. 회황색을 띰\n2. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n3. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n4. 산조인에 비해 두께가 얇고 납작함'
only_edge_ZIMA = '1. 산조인에 비해 두께가 얇고 납작함\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n4. 회황색을 띰'

print()