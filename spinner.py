import cv2
import mediapipe as mp
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()

# Create a VideoCapture object to read frames from the camera
cap = cv2.VideoCapture(0)

# Create a MediaPipe Hands object and configure the drawing settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# Set the background color to black
bg_color = (0, 0, 0)  # Black color

# Set the hand landmarks and finger connections color to white
landmark_color = (255, 255, 255)  # White color
connection_color = (255, 255, 255)  # White color
pygame.init()
display = (800,600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

glTranslatef(0.0,0.0, -5)
# Start the video processing loop
while True:
    # Read frame from camera
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror-like effect
    image = cv2.flip(image, 1)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)

    # Clear the background by filling it with the bg_color
    image = cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), bg_color, -1)

    # Draw hand landmarks and finger connections on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=connection_color, thickness=2, circle_radius=2),
            )

            # Calculate distance between index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            x_diff = -thumb_tip.x + index_finger_tip.x
            y_diff = -thumb_tip.y + index_finger_tip.y
            z_diff = -thumb_tip.z + index_finger_tip.z

            distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        glRotatef(x_diff*100, y_diff*100, z_diff*100, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)

    # Display the image
    cv2.imshow("Hand Tracking", image)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the windows
cap.release()
cv2.destroyAllWindows()
