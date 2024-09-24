# hand_recognition.py

import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Finger names according to MediaPipe landmarks
    finger_names = ["Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Pinky"]

    # MediaPipe hand landmark indices for each finger tip
    finger_tips_ids = [4, 8, 12, 16, 20]

    while True:
        # Capture frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            break

        # Flip the image horizontally for natural hand movements
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand detection
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the hand
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract finger tips landmarks
                for i, tip_id in enumerate(finger_tips_ids):
                    x = int(hand_landmarks.landmark[tip_id].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[tip_id].y * frame.shape[0])
                    
                    # Label the fingers
                    cv2.putText(frame, finger_names[i], (x - 30, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Hand Tracking with Finger Names", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
