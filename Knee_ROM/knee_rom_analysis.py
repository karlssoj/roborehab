from ultralytics import YOLO
import cv2
import math
import datetime
import collections
import statistics

model = YOLO('Yolo11_cvrehab_lite.pt')  # load an official model

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("knee_bending.mp4")

measuring = False
left = False
side_selected = False

fNum = 0

MEDIAN_WINDOW = 7  # Number of frames for median smoothing
angle_history = collections.deque(maxlen=MEDIAN_WINDOW)

def get_distance(pos1, pos2):
    dist = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    return dist


def choose_side_with_buttons(event, x, y, flags, param):
    global left, side_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get frame dimensions from param
        frame_width = param['width']
        frame_height = param['height']
        left_button_center = [frame_width - 100, 50]
        right_button_center = [100, 50]
        save_button_center = [frame_width // 2, frame_height - 50]

        if get_distance([x, y], left_button_center) < 40:
            left = True
            side_selected = True
            print("Left side selected")
        elif get_distance([x, y], right_button_center) < 40:
            left = False
            side_selected = True
            print("Right side selected")
        elif get_distance([x, y], save_button_center) < 50:
            # Save button clicked
            if param.get('measuring') and param.get('knee_string'):
                save_value(param['knee_string'], './Measurements/Measurements.txt')
                print("Measurement saved!")


def draw_side_buttons(frame, show_save_button=False):
    # Left button (top right)
    left_button_center = [frame.shape[1] - 100, 50]
    cv2.circle(frame, left_button_center, 40, (0, 255, 0) if left and side_selected else (255, 255, 255), -1 if left and side_selected else 3)
    cv2.putText(frame, 'L', (left_button_center[0] - 15, left_button_center[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)

    # Right button (top left)
    right_button_center = [100, 50]
    cv2.circle(frame, right_button_center, 40, (0, 255, 0) if not left and side_selected else (255, 255, 255), -1 if not left and side_selected else 3)
    cv2.putText(frame, 'R', (right_button_center[0] - 15, right_button_center[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Save button (bottom center) - only show when measuring
    if show_save_button:
        save_button_center = [frame.shape[1] // 2, frame.shape[0] - 50]
        cv2.circle(frame, save_button_center, 50, (0, 200, 255), -1)
        cv2.putText(frame, 'SAVE', (save_button_center[0] - 45, save_button_center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


def calculate_knee_angle(hip, knee, ankle):
    """Calculate knee angle where 0 degrees = straight leg, >0 = bent leg"""
    # Vector from knee to hip
    vector1 = (hip[0] - knee[0], hip[1] - knee[1])
    # Vector from knee to ankle
    vector2 = (ankle[0] - knee[0], ankle[1] - knee[1])

    # Calculate dot product and magnitudes
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    # Calculate angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)

    # Return angle where straight = 0, bent = positive
    return 180 - angle_degrees


def save_value(value, path):
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    save_data = timestamp + ': ' + value + '\n'

    with open(path, 'a') as file:
        file.write(save_data)

def draw_knee_joints(frame, hip, knee, ankle):
    global measuring
    global angle

    if knee[0] != 0 and ankle[0] != 0 and hip[0] != 0:
        # Draw lines connecting joints
        cv2.line(frame, hip, knee, (255, 0, 0), 3)
        cv2.line(frame, knee, ankle, (255, 0, 0), 3)

        # Draw joint circles
        cv2.circle(frame, hip, 5, (0, 255, 0), -1)
        cv2.circle(frame, knee, 5, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 5, (255, 0, 0), -1)

        # Calculate knee angle
        angle = calculate_knee_angle(hip, knee, ankle)
        angle_history.append(angle)
        measuring = True
    else:
        measuring = False
        angle = 0

    return angle




# Create window and set mouse callback
cv2.namedWindow('YOLOv11-based Knee ROM Analysis')
callback_set = False

while cap.isOpened():
    ret, frame = cap.read()

    measuring = False

    if not ret:
        break

    frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_NEAREST)

    # Set mouse callback once with frame dimensions
    if not callback_set:
        cv2.setMouseCallback('YOLOv11-based Knee ROM Analysis', choose_side_with_buttons, {'width': frame.shape[1], 'height': frame.shape[0]})
        callback_set = True

    draw_side_buttons(frame, measuring)

    # Predict with the model
    results = model(frame)

    for r in results:
        keypoints = r.keypoints.xy.int().numpy()

        # Check if any person is detected
        if len(keypoints) == 0:
            continue

        left_hip = keypoints[0][11]
        right_hip = keypoints[0][12]
        left_knee = keypoints[0][13]
        right_knee = keypoints[0][14]
        left_ankle = keypoints[0][15]
        right_ankle = keypoints[0][16]

        angle = 0

        if side_selected:
            if left:
                # Show only left leg joints
                if left_knee[0] != 0 and left_ankle[0] != 0 and left_hip[0] != 0:
                    angle = draw_knee_joints(frame, left_hip, left_knee, left_ankle)
            else:
                # Show only right leg joints
                if right_knee[0] != 0 and right_ankle[0] != 0 and right_hip[0] != 0:
                    angle = draw_knee_joints(frame, right_hip, right_knee, right_ankle)

    knee_string = ''

    # Display selected side
    if side_selected:
        side_text = 'Left side' if left else 'Right side'
        cv2.putText(frame, side_text, (frame.shape[1]//2 - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if measuring:
        if left:
            knee_string = 'Left knee angle = '
        else:
            knee_string = 'Right knee angle = '

        median_angle = statistics.median(angle_history) if angle_history else angle
        knee_string = knee_string + str(int(median_angle)) + ' degrees'
        cv2.putText(frame, knee_string, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    elif not side_selected:
        cv2.putText(frame, 'Click L or R button to select side', (frame.shape[1]//2 - 250, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Update mouse callback with current measuring state and knee_string
    cv2.setMouseCallback('YOLOv11-based Knee ROM Analysis', choose_side_with_buttons, 
                        {'width': frame.shape[1], 'height': frame.shape[0], 'measuring': measuring, 'knee_string': knee_string})

    cv2.imshow('YOLOv11-based Knee ROM Analysis', frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

    # Save with Enter or S key
    if key == 13 or key == ord('s') or key == ord('S'):
        if measuring:
            save_value(knee_string, './Measurements/Measurements.txt')
            print("Measurement saved!")

    # Select left side with L key
    if key == ord('l') or key == ord('L'):
        left = True
        side_selected = True
        print("Left side selected")

    # Select right side with R key
    if key == ord('r') or key == ord('R'):
        left = False
        side_selected = True
        print("Right side selected")

cap.release()
cv2.destroyAllWindows()