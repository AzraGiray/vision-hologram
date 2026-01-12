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

RED = (220, 50, 50)
WHITE = (255, 255, 255)

# -------------------- MEDIAPIPE SETUP --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

# -------------------- STATE --------------------
cube_size = 80
rotation_angle = 0

active_pos = [560, 300]     # sağ el ile taşınan küp
fixed_cubes = []            # sabitlenen küpler

is_left_fist = False
prev_left_fist = False

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

    # frame başı reset
    is_left_fist = False

    result = hands.process(rgb)

    # -------------------- HAND PROCESSING --------------------
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            label = hand_info.classification[0].label.upper()
            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)

            # -------- RIGHT HAND → MOVE --------
            if label == "RIGHT":
                active_pos[0] = x
                active_pos[1] = y

            # -------- LEFT HAND → FIST (DROP) --------
            if label == "LEFT":
                index_mcp = hand_landmarks.landmark[5]
                fist_distance = math.hypot(
                    index_tip.x - index_mcp.x,
                    index_tip.y - index_mcp.y
                )
                is_left_fist = fist_distance < 0.05

    # -------------------- DROP EDGE DETECTION --------------------
    just_dropped = is_left_fist and not prev_left_fist
    prev_left_fist = is_left_fist

    if just_dropped:
        fixed_cubes.append({
            "pos": tuple(active_pos),
            "size": cube_size,
            "angle": rotation_angle
        })
        print(f"Cube dropped. Total fixed cubes: {len(fixed_cubes)}")

    # -------------------- DRAW CAMERA --------------------
    cam_surface = pygame.surfarray.make_surface(np.rot90(rgb))
    screen.blit(cam_surface, (0, 0))

    # -------------------- DRAW FIXED CUBES --------------------
    for cube in fixed_cubes:
        surface = pygame.Surface((cube["size"], cube["size"]), pygame.SRCALPHA)
        surface.fill(RED)
        rotated = pygame.transform.rotate(surface, cube["angle"])
        rect = rotated.get_rect(center=cube["pos"])
        screen.blit(rotated, rect)

    # -------------------- DRAW ACTIVE CUBE --------------------
    cube_surface = pygame.Surface((cube_size, cube_size), pygame.SRCALPHA)
    cube_surface.fill(RED)
    rotated_cube = pygame.transform.rotate(cube_surface, rotation_angle)
    rect = rotated_cube.get_rect(center=active_pos)
    screen.blit(rotated_cube, rect)

    # -------------------- DEBUG TEXT --------------------
    font = pygame.font.SysFont(None, 36)
    status = "LEFT FIST" if is_left_fist else "LEFT OPEN"
    text_surface = font.render(status, True, WHITE)
    screen.blit(text_surface, (20, 20))

    pygame.display.flip()
    clock.tick(30)
