"""
================================================================================================
Name			: protractor.py
Author			: Wonho Son (sonwonho2005@gmail.com)
Last modified	: 2022-11-20
Description		: 내시경 각도 측정 프로그램
================================================================================================
"""


import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import nonzero
from numpy.core.numeric import Inf, NaN


def cal_degree(frame, mask):
    """이미지와 마스크를 이용하여 각도 계산

    Args:
        frame (array): 이미지
        mask (array): 마스크
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return
    perimeter = []
    for cnt in contours:
        perimeter.append(cv2.arcLength(cnt, True))
    if perimeter:
        maxindex = perimeter.index(max(perimeter))
    else:
        maxindex = -1

    epsilon = 0.018 * cv2.arcLength(contours[maxindex], True)
    print(epsilon)
    approx = cv2.approxPolyDP(contours[maxindex], epsilon, True)
    deg = 0

    if len(approx) >= 3:
        distances = []
        for i in range(len(approx) - 1):
            x1 = approx[i][0][1]
            y1 = approx[i][0][0]
            x2 = approx[i + 1][0][1]
            y2 = approx[i + 1][0][0]
            w = x1 - x2
            h = y1 - y2
            distance = math.sqrt((w * w) + (h * h))
            distances.append(distance)
        x1 = approx[-1][0][1]
        y1 = approx[-1][0][0]
        x2 = approx[0][0][1]
        y2 = approx[0][0][0]
        w = x1 - x2
        h = y1 - y2
        distance = math.sqrt((w * w) + (h * h))
        distances.append(distance)
        minDistanceIndex = distances.index(sorted(distances)[0])
        minDistanceIndex2 = distances.index(sorted(distances)[1])

        if minDistanceIndex == len(approx) - 1:
            a = [approx[minDistanceIndex][0][0], approx[minDistanceIndex][0][1]]
            b = [approx[0][0][0], approx[0][0][1]]
            x1 = int((approx[minDistanceIndex][0][0] + approx[0][0][0]) / 2)
            y1 = int((approx[minDistanceIndex][0][1] + approx[0][0][1]) / 2)
        else:
            a = [approx[minDistanceIndex][0][0], approx[minDistanceIndex][0][1]]
            b = [approx[minDistanceIndex + 1][0][0], approx[minDistanceIndex + 1][0][1]]
            x1 = int((approx[minDistanceIndex][0][0] + approx[minDistanceIndex + 1][0][0]) / 2)
            y1 = int((approx[minDistanceIndex][0][1] + approx[minDistanceIndex + 1][0][1]) / 2)
        if minDistanceIndex2 == len(approx) - 1:
            c = [approx[minDistanceIndex2][0][0], approx[minDistanceIndex2][0][1]]
            d = [approx[0][0][0], approx[0][0][1]]
            x2 = int((approx[minDistanceIndex2][0][0] + approx[0][0][0]) / 2)
            y2 = int((approx[minDistanceIndex2][0][1] + approx[0][0][1]) / 2)
        else:
            c = [approx[minDistanceIndex2][0][0], approx[minDistanceIndex2][0][1]]
            d = [approx[minDistanceIndex2 + 1][0][0], approx[minDistanceIndex2 + 1][0][1]]
            x2 = int((approx[minDistanceIndex2][0][0] + approx[minDistanceIndex2 + 1][0][0]) / 2)
            y2 = int((approx[minDistanceIndex2][0][1] + approx[minDistanceIndex2 + 1][0][1]) / 2)

        m1 = find_orthogonal_inclination(a, b)
        m2 = find_orthogonal_inclination(c, d)

        if y1 <= y2:
            start = [x1, y1]
            end = [x2, y2]
            y_start = find_y_intercept(m1, start)
            y_end = find_y_intercept(m2, end)
            interserction_point = find_interserction_point(m1, m2, y_start, y_end)

        else:
            start = [x2, y2]
            end = [x1, y1]
            y_start = find_y_intercept(m2, start)
            y_end = find_y_intercept(m1, end)
            interserction_point = find_interserction_point(m2, m1, y_start, y_end)

        print(m1, m2, y_start, y_end, interserction_point)
        if (
            math.isnan(interserction_point[0])
            or math.isnan(interserction_point[1])
            or abs(interserction_point[0]) == math.inf
            or abs(interserction_point[1]) == math.inf
            or m1 == math.inf
            or m2 == math.inf
        ):
            return 0
        else:
            interserction_point[0] = int(interserction_point[0])
            interserction_point[1] = int(interserction_point[1])

        deg = getAngle(start, interserction_point, end)

        if end[1] < interserction_point[1]:
            if deg > 0:
                deg = deg - 360
            else:
                deg = 360 + deg

        cv2.drawContours(frame, [contours[maxindex]], 0, (255, 255, 0), 2)
        cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)

        cv2.line(frame, a, b, (0, 255, 0), 2)
        cv2.line(frame, c, d, (0, 255, 0), 2)
        cv2.line(frame, start, interserction_point, (0, 0, 255), 2)
        cv2.line(frame, end, interserction_point, (0, 0, 255), 2)
        cv2.circle(frame, interserction_point, 5, (0, 0, 255), -1)
        cv2.putText(frame, "start", tuple(start), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, "end", tuple(end), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(
            frame,
            "{0:.2f}".format(deg),
            interserction_point,
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame, "{0:.2f}".format(deg), (30, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1
        )

        return deg


def color_filter(frame):
    """색을 필터링하여 마스크를 획득하는 함수

    Args:
        frame (array): 이미지

    Returns:
        black_mask (array): 필터링된 마스크
        binary (array): 필터링된 마스크를 임계 처리하여 이진화한 마스크
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([360, 255, 60])

    black_mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), dtype=np.float64)
    mask = cv2.filter2D(black_mask.copy(), -1, kernel)
    _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    return black_mask, binary


def find_orthogonal_inclination(a, b):
    """a, b 두 점을 지나는 그래프와 직교를 이루는 직선의 기울기를 구하는 함수

    Args:
        a (list): a 점의 (x, y) 좌표
        b (list): b 점의 (x, y) 좌표

    Returns:
        float: a, b 두 점을 지나는 그래프와 직교를 이루는 직선의 기울기
    """
    x1 = a[0]
    x2 = b[0]
    y1 = a[1]
    y2 = b[1]
    inclination = (y2 - y1) / (x2 - x1)
    return -1.0 / inclination


def find_y_intercept(inclination, point):
    """주어진 기울기와 좌표를 지나는 1차 그래프의 y 절편을 구하는 함수

    Args:
        inclination (float): 기울기
        point (list): x, y 좌표

    Returns:
        float: y 절편
    """
    x, y = point
    y_intercept = y - (inclination * x)
    return y_intercept


def find_interserction_point(inclination1, inclination2, y_intercept1, y_intercept2):
    """주어진 기울기와 y 절편을 가지는 두 그래프의 교점을 구하는 함수

    Args:
        inclination1 (float): 1번 그래프의 기울기
        inclination2 (float): 2번 그래프의 기울기
        y_intercept1 (float): 1번 그래프의 y 절편
        y_intercept2 (float): 2번 그래프의 y 절편

    Returns:
        list: 두 그래프의 교점 좌표
    """
    x = -1 * (y_intercept1 - y_intercept2) / (inclination1 - inclination2)
    y = (inclination1 * x) + y_intercept1
    return [x, y]


def getAngle(a, b, c):
    """a, b, c 세 점이 이루는 각도를 계산하는 함수

    Args:
        a (list): a 점의 (x, y) 좌표
        b (list): b 점의 (x, y) 좌표
        c (list): c 점의 (x, y) 좌표

    Returns:
        int: 각도
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    ang = ang + 360 if ang < 0 else ang
    return ang - 180


if __name__ == "__main__":
    import os

    margin = 10
    for i in range(1, 14):
        ori_frame = cv2.imread(os.path.join("images", "{}.jpg").format(i + 1))
        ratio = 600.0 / ori_frame.shape[1]
        dim = (600, int(ori_frame.shape[0] * ratio))
        ori_frame = cv2.resize(ori_frame, dim)
        for j in range(3):
            frame = ori_frame.copy()
            if j < 2:
                frame = cv2.flip(frame, j)
            flip_frame = frame
            mask, mask2 = color_filter(frame)

            cal_degree(frame, mask2)

            cv2.imshow("ori_frame", ori_frame)
            cv2.imshow("frame", frame)
            cv2.imshow(
                "maskframe", cv2.multiply(frame, np.multiply(np.dstack((mask2, mask2, mask2)), 255))
            )
            cv2.imshow("mask", mask)
            cv2.imshow("mask2", mask2)
            cv2.waitKey(0)
