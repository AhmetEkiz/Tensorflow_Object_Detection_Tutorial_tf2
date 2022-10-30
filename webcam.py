import cv2
import time

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# Setup capture
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if __name__ == '__main__':
	while True:

		new_frame_time = time.time()
		fps = 1 / (new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time

		ret, frame = cap.read()

		cv2.imshow('webcam', cv2.resize(frame, (800, 600)))


		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			break

		# fps = int(fps)
		fps = str(fps)
		print(fps)
		