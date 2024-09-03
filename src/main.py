import cv2
import numpy as np

class OpticalFlowDetector:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=10, blockSize=7)
        self.trajectory_len = 40
        self.detect_interval = 5
        self.trajectories = []
        self.frame_idx = 0
        self.prev_gray = None

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()

        if len(self.trajectories) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []
            for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                trajectory.append((x, y))
                if len(trajectory) > self.trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            self.trajectories = new_trajectories

        if self.frame_idx % self.detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(trajectory[-1]) for trajectory in self.trajectories]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.trajectories.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray
        return img