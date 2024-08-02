import cv2
import numpy as np
import cupy as cp
from collections import deque
import matplotlib.pyplot as plt

class CameraTamperingDetector:
    def __init__(self, short_term_size=3, long_term_size=36):
        self.short_term_pool = deque(maxlen=short_term_size)
        self.long_term_pool = deque(maxlen=long_term_size)
        self.frame_count = 0

    def compute_chromaticity_histogram(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def compute_dissimilarity(self, hist1, hist2):
        return np.sum(np.abs(hist1 - hist2))

    def compute_dissimilarity_vectorized(self, pool1, pool2):
        pool1 = cp.array(pool1)
        pool2 = cp.array(pool2)
        batch_size = 100  # Adjust batch size to avoid Out of Memory error
        diffs = []
        for i in range(0, len(pool1), batch_size):
            for j in range(0, len(pool2), batch_size):
                batch_diffs = cp.abs(pool1[i:i + batch_size, cp.newaxis, :] - pool2[cp.newaxis, j:j + batch_size, :])
                diffs.append(cp.sum(batch_diffs, axis=2).flatten())
        return cp.concatenate(diffs)

    def detect_tampering(self, frame):
        self.frame_count += 1
        hist = self.compute_chromaticity_histogram(frame)
        
        self.short_term_pool.append(hist)
        if len(self.short_term_pool) == self.short_term_pool.maxlen:
            self.long_term_pool.append(self.short_term_pool[0])

        if len(self.long_term_pool) < self.long_term_pool.maxlen:
            return False, 0.0

        short_term_diffs = cp.asnumpy(self.compute_dissimilarity_vectorized(self.short_term_pool, self.long_term_pool))
        long_term_diffs = cp.asnumpy(self.compute_dissimilarity_vectorized(self.long_term_pool, self.long_term_pool))

        d_between = np.median(short_term_diffs)
        d_long = np.median(long_term_diffs)

        d_norm = np.log(d_between / d_long)

        threshold = 1
        print(d_norm)
        return d_norm > threshold, d_norm

def process_webcam():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    short_term_size = int(3 * 5)
    long_term_size = int(3 * 60)

    print(f"short_term_size: {short_term_size}, long_term_size: {long_term_size}")

    detector = CameraTamperingDetector(short_term_size=short_term_size, long_term_size=long_term_size)

    d_norm_values = []

    frame_interval = int(fps / 3)

    while True:
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            break

        tampering_detected, d_norm = detector.detect_tampering(frame)
        d_norm_values.append(d_norm)

        if detector.frame_count >= detector.long_term_pool.maxlen:
            if tampering_detected:
                cv2.putText(frame, "TAMPERING DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Initializing... {detector.frame_count}/{detector.long_term_pool.maxlen}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        fig, ax = plt.subplots(figsize=(3, 6))
        ax.plot(d_norm_values[-100:], range(len(d_norm_values[-100:])))
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 100)
        ax.set_xlabel('d_norm')
        ax.set_ylabel('Frame')
        ax.set_title('Dissimilarity')

        plt.tight_layout()
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        plot_img = cv2.resize(plot_img, (300, frame_height))

        combined_frame = np.hstack((frame, plot_img[:, :, :3]))

        cv2.imshow('Combined Feed', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()
