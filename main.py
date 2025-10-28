import cv2
import mediapipe as mp
import pyautogui
import time

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(3, screen_w)
cap.set(4, screen_h)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

ema_x, ema_y = 0, 0
alpha = 0.3
cooldown = 0.8
last_action_time = 0
cursor_locked = False

locked_position = (0, 0)

hand_present = False
cursor_active = False

def is_finger_down(lm, tip, pip, threshold=0.04):
    return lm[tip].y > lm[pip].y + threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_present = True
        for hand in results.multi_hand_landmarks:
            lm = hand.landmark
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            pinky_down = is_finger_down(lm, 20, 18)
            index_down = is_finger_down(lm, 8, 6)
            middle_down = is_finger_down(lm, 12, 10)
            ring_down = is_finger_down(lm, 16, 14)
            now = time.time()

            ix, iy = lm[8].x * screen_w, lm[8].y * screen_h
            mx, my = lm[12].x * screen_w, lm[12].y * screen_h
            mid_x, mid_y = (ix + mx) / 2, (iy + my) / 2

            if not cursor_active:
                ema_x, ema_y = pyautogui.position()
                cursor_active = True

            if pinky_down and now - last_action_time > cooldown:
                cursor_locked = not cursor_locked
                last_action_time = now
                if cursor_locked:
                    locked_position = (ema_x, ema_y)
                    cv2.putText(frame, "Cursor Locked", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "Cursor Unlocked", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

            if not cursor_locked:
                ema_x = (1 - alpha) * ema_x + alpha * mid_x
                ema_y = (1 - alpha) * ema_y + alpha * mid_y

                ema_x = min(max(0, ema_x), screen_w - 1)
                ema_y = min(max(0, ema_y), screen_h - 1)

                pyautogui.moveTo(ema_x, ema_y, duration=0.01)
            else:
                pyautogui.moveTo(locked_position[0], locked_position[1], duration=0.01)

            if index_down and not middle_down and not ring_down and now - last_action_time > cooldown:
                pyautogui.click(button='left')
                last_action_time = now
                cv2.putText(frame, "Left Click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if middle_down and not index_down and not ring_down and now - last_action_time > cooldown:
                pyautogui.click(button='right')
                last_action_time = now
                cv2.putText(frame, "Right Click", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if index_down and middle_down and not ring_down and now - last_action_time > cooldown:
                pyautogui.scroll(-40)
                last_action_time = now
                cv2.putText(frame, "Scroll Down", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            if index_down and middle_down and ring_down and now - last_action_time > cooldown:
                pyautogui.scroll(40)
                last_action_time = now
                cv2.putText(frame, "Scroll Up", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 3)


    else:
        if hand_present:
            hand_present = False
            cursor_active = False   


    cv2.imshow("Gesture Mouse Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
