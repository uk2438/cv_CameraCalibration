import cv2 as cv
import numpy as np
import os

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, pts = cv.findChessboardCorners(gray, board_pattern)
        if ret:
            img_points.append(pts)
    assert len(img_points) > 0, "체커보드가 감지된 프레임이 없습니다!"

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)


# --- 설정 ---
video_name = "chessboard.avi"
board_pattern = (10, 7)
board_cellsize = 25.0
output_folder = "undistorted_results"
os.makedirs(output_folder, exist_ok=True)

# --- 비디오 열기 ---
video = cv.VideoCapture(video_name)
video_images = []

if not video.isOpened():
    print("비디오를 열 수 없습니다")
    exit()

# --- 체커보드 감지된 프레임만 수집 ---
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, board_pattern)
    if found:
        video_images.append(frame)

video.release()

# --- 카메라 캘리브레이션 ---
if len(video_images) > 0:
    ret, K, dist_coeffs, rvecs, tvecs = calib_camera_from_chessboard(
        video_images, board_pattern, board_cellsize
    )

    print(f"재투영 오차: {ret}")
    print(f"K:\n{K}")
    print(f"왜곡 계수:\n{dist_coeffs}")

    # 🎞️ 캘리브레이션 프레임 영상으로 보여주기
    for frame in video_images:
        cv.imshow("Calibration Frames (Original)", frame)
        if cv.waitKey(30) & 0xFF == 27:  # ESC 키로 중단
            break
    cv.destroyAllWindows()

    # 💾 왜곡 보정된 프레임 저장
    for i, img in enumerate(video_images):
        undistorted = cv.undistort(img, K, dist_coeffs)
        filename = os.path.join(output_folder, f"undistorted_{i:03}.png")
        cv.imwrite(filename, undistorted)

    print(f"왜곡 보정된 이미지 {len(video_images)}개를 '{output_folder}' 폴더에 저장했습니다.")

else:
    print("체커보드가 감지된 프레임이 없습니다.")