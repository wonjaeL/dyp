import csv
import os
import platform
import subprocess

import cv2
import numpy as np


def is_windows():
    """현재 플랫폼이 Windows인지 확인합니다."""
    return platform.system().lower() == 'windows'


class HandTrajectoryRunner:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        self.trajectory_points = []
        self.lower_skin_hsv = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin_hsv = np.array([15, 255, 255], dtype=np.uint8)

    def create_skin_mask(self, frame):
        """주어진 프레임에서 피부색 마스크를 생성합니다."""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv_frame, self.lower_skin_hsv, self.upper_skin_hsv)

    def extract_contours(self, frame):
        """프레임에서 윤곽선을 추출합니다."""
        skin = self.create_skin_mask(frame)
        blurred_skin = cv2.GaussianBlur(skin, (7, 7), 0)
        contours, _ = cv2.findContours(blurred_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_largest_contour(self, contours):
        """윤곽선 중 가장 큰 면적을 가진 윤곽선을 찾습니다."""
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
        return largest_contour

    def calculate_center(self, contour):
        """주어진 윤곽선의 중심점을 계산합니다."""
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None

    def draw_path(self, frame):
        """프레임에 손의 궤적을 그립니다."""
        for i in range(1, len(self.trajectory_points)):
            cv2.line(frame, self.trajectory_points[i - 1], self.trajectory_points[i], (0, 255, 0), 2)

    def process_contour(self, frame, contour):
        """윤곽선과 중심점을 처리합니다."""
        center = self.calculate_center(contour)
        if center is not None:
            self.trajectory_points.append(center)
            cv2.drawContours(frame, [contour], 0, (255, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    def draw_minimap(self, frame):
        """프레임의 우측 하단에 미니맵을 그립니다."""
        minimap_size = 100
        minimap = np.zeros((minimap_size, minimap_size, 3), dtype=np.uint8)
        for i in range(1, len(self.trajectory_points)):
            x1, y1 = self.trajectory_points[i - 1]
            x2, y2 = self.trajectory_points[i]
            x1, y1 = int(x1 * minimap_size / frame.shape[1]), int(y1 * minimap_size / frame.shape[0])
            x2, y2 = int(x2 * minimap_size / frame.shape[1]), int(y2 * minimap_size / frame.shape[0])
            cv2.line(minimap, (x1, y1), (x2, y2), (0, 255, 0), 1)
        frame[frame.shape[0] - minimap_size:, frame.shape[1] - minimap_size:] = minimap

    def get_movement_direction(self):
        """손의 현재 움직임 방향을 구합니다."""
        if len(self.trajectory_points) < 2:
            return "None"
        x1, y1 = self.trajectory_points[-2]
        x2, y2 = self.trajectory_points[-1]
        dx, dy = x2 - x1, y2 - y1

        if abs(dx) > abs(dy):
            return "Right" if dx > 0 else "Left"
        else:
            return "Up" if dy < 0 else "Down"

    def display_frame(self, frame):
        """프레임에 미니맵과 움직임 방향을 그리고 화면에 표시합니다."""
        self.draw_minimap(frame)
        direction = self.get_movement_direction()
        cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Hand Trajectory", frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            return False
        return True

    def run(self):
        """손의 궤적을 추적하고 표시하는 프로세스를 실행합니다."""
        if not self.video.isOpened():
            print("Error: Could not open the video.")
            exit()

        for ret, frame in iter(lambda: self.video.read(), (False, None)):
            contours = self.extract_contours(frame)
            largest_contour = self.find_largest_contour(contours)
            if largest_contour is not None:
                self.process_contour(frame, largest_contour)
            self.draw_path(frame)
            if not self.display_frame(frame):
                break

        self.video.release()
        cv2.destroyAllWindows()

    def save_trajectory_to_csv(self, file_name):
        """손의 궤적 데이터를 CSV 파일로 저장합니다."""
        with open(file_name, 'w', newline='') as csvfile:
            fieldnames = ['time', 'x', 'y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for time, (x, y) in enumerate(self.trajectory_points):
                writer.writerow({'time': time, 'x': x, 'y': y})


if __name__ == "__main__":
    runner = HandTrajectoryRunner('seungbo.mp4')
    runner.run()

    file_name = 'trajectory_data.csv'
    runner.save_trajectory_to_csv(file_name)

    if is_windows():
        os.startfile(file_name)
    else:
        subprocess.call(['open', file_name])
