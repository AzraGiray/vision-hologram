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
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

# -------------------- STATE VARIABLES --------------------
cube_size = 80
rotation_angle = 0

is_fist = False
prev_is_fist = False
fixed_cubes = []
cube_locked_this_fist = False


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

    # reset each frame
    is_fist = False

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # ---------- LANDMARKS ----------
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        index_mcp = hand_landmarks.landmark[5]

        # ---------- SCALE (PINCH) ----------
        pinch_distance = math.hypot(
            (thumb_tip.x - index_tip.x) * WIDTH,
            (thumb_tip.y - index_tip.y) * HEIGHT
        )

        scale = int(pinch_distance / 4)
        cube_size = max(40, min(160, scale))

        # ---------- ROTATE ----------
        x1, y1 = int(index_mcp.x * WIDTH), int(index_mcp.y * HEIGHT)
        x2, y2 = int(index_tip.x * WIDTH), int(index_tip.y * HEIGHT)

        dx = x2 - x1
        dy = y1 - y2  # screen coord fix

        rotation_angle = math.degrees(math.atan2(dy, dx))

        # ---------- FIST DETECTION ----------
        fist_distance = math.hypot(
            index_tip.x - index_mcp.x,
            index_tip.y - index_mcp.y
        )

        is_fist = fist_distance < 0.05

    # ---------- EDGE DETECTION (GLOBAL) ----------
    just_closed = is_fist and not prev_is_fist
    prev_is_fist = is_fist
    
    if not is_fist:
        cube_locked_this_fist = False

    if just_closed and not cube_locked_this_fist:
        fixed_cubes.append({
            "pos": (560, 300),
            "size": cube_size,
            "angle": rotation_angle
        })
        cube_locked_this_fist = True
        print(f"Cube locked. Total fixed cubes: {len(fixed_cubes)}")

 


    # -------------------- DRAW CAMERA --------------------
    cam_surface = pygame.surfarray.make_surface(np.rot90(rgb))
    screen.blit(cam_surface, (0, 0))
    
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
    rect = rotated_cube.get_rect(center=(560, 300))
    screen.blit(rotated_cube, rect)

    # -------------------- DEBUG TEXT --------------------
    font = pygame.font.SysFont(None, 36)
    status_text = "FIST" if is_fist else "OPEN"
    text_surface = font.render(status_text, True, WHITE)
    screen.blit(text_surface, (20, 20))

    pygame.display.flip()
    clock.tick(30)
