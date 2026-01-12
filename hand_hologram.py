import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np
import math

# -------------------- PYGAME SETUP --------------------
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Thought → Hologram Demo")
clock = pygame.time.Clock()

BLUE = (50, 150, 255)
RED = (220, 50, 50)

# -------------------- MEDIAPIPE SETUP --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
sphere_radius = 40
cube_size = 80
rotation_angle = 0


# -------------------- MAIN LOOP --------------------
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(thumb_tip.x * WIDTH), int(thumb_tip.y * HEIGHT)
        x2, y2 = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)

        distance = math.hypot(x2 - x1, y2 - y1)

        # Ölçekleme (deneysel ama stabil)
        scale = int(distance / 4)

        sphere_radius = max(20, min(100, scale))
        cube_size = max(40, min(160, scale))

        index_mcp = hand_landmarks.landmark[5]
        index_tip = hand_landmarks.landmark[8]

        x1, y1 = int(index_mcp.x * WIDTH), int(index_mcp.y * HEIGHT)
        x2, y2 = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)

        dx = x2 - x1
        dy = y1 - y2  # ekran koordinatı ters olduğu için

        rotation_angle = math.degrees(math.atan2(dy, dx))

    detected_hands = []


    detected_hands.clear()

    if result.multi_handedness:
        for hand_info in result.multi_handedness:
            label = hand_info.classification[0].label.upper()
            detected_hands.append(label)


    # -------------------- DRAW CAMERA --------------------
    cam_surface = pygame.surfarray.make_surface(np.rot90(rgb))
    screen.blit(cam_surface, (0, 0))

    # -------------------- DRAW HOLOGRAM --------------------
    if "LEFT" in detected_hands:
        cube_surface = pygame.Surface((cube_size, cube_size), pygame.SRCALPHA)
        cube_surface.fill(RED)
        rotated_cube = pygame.transform.rotate(cube_surface, rotation_angle)
        rect = rotated_cube.get_rect(center=(560, 300))
        screen.blit(rotated_cube, rect)



    if "RIGHT" in detected_hands:
        pygame.draw.circle(screen, BLUE, (200, 300), sphere_radius)



    pygame.display.flip()
    clock.tick(30)
