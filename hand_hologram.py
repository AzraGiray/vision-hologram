import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np
import math

# ==================== 3D CUBE DRAW ====================
GREEN_FRONT = (0, 220, 120)
GREEN_TOP   = (60, 255, 170)
GREEN_SIDE  = (0, 160, 90)
WHITE = (255, 255, 255)

def draw_3d_cube(surface, center, size, depth, filled=True):
    cx, cy = center
    d = int(depth)

    front = pygame.Rect(
        cx - size // 2,
        cy - size // 2,
        size,
        size
    )

    top_face = [
        (front.left, front.top),
        (front.right, front.top),
        (front.right - d, front.top - d),
        (front.left - d, front.top - d)
    ]

    side_face = [
        (front.right, front.top),
        (front.right, front.bottom),
        (front.right - d, front.bottom - d),
        (front.right - d, front.top - d)
    ]

    if filled:
        pygame.draw.polygon(surface, GREEN_TOP, top_face)
        pygame.draw.polygon(surface, GREEN_SIDE, side_face)
        pygame.draw.rect(surface, GREEN_FRONT, front)
    else:
        pygame.draw.polygon(surface, GREEN_TOP, top_face, 2)
        pygame.draw.polygon(surface, GREEN_SIDE, side_face, 2)
        pygame.draw.rect(surface, GREEN_FRONT, front, 2)

# ==================== PYGAME ====================
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Thought → Hologram Demo")
clock = pygame.time.Clock()

# ==================== MEDIAPIPE ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ==================== STATE ====================
cube_size = 80
depth = 20

active_pos = [560, 300]     # sol el ile taşınan küp
fixed_cubes = []            # sabitlenen küpler

is_right_fist = False
prev_right_fist = False

# ==================== MAIN LOOP ====================
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

    is_right_fist = False

    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            label = hand_info.classification[0].label.upper()
            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)

            # LEFT HAND → MOVE + DEPTH
            if label == "LEFT":
                active_pos[:] = [x, y]

                thumb_tip = hand_landmarks.landmark[4]
                pinch_distance = math.hypot(
                    (thumb_tip.x - index_tip.x) * WIDTH,
                    (thumb_tip.y - index_tip.y) * HEIGHT
                )
                depth = max(10, min(40, int(pinch_distance / 8)))

            # RIGHT HAND → DROP
            if label == "RIGHT":
                index_mcp = hand_landmarks.landmark[5]
                fist_distance = math.hypot(
                    index_tip.x - index_mcp.x,
                    index_tip.y - index_mcp.y
                )
                is_right_fist = fist_distance < 0.05

    just_dropped = is_right_fist and not prev_right_fist
    prev_right_fist = is_right_fist

    if just_dropped:
        fixed_cubes.append({
            "pos": tuple(active_pos),
            "size": cube_size,
            "depth": depth
        })
        print(f"Cube dropped. Total fixed cubes: {len(fixed_cubes)}")

    # ==================== DRAW ====================
    cam_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    screen.blit(cam_surface, (0, 0))

    # fixed cubes
    for cube in fixed_cubes:
        draw_3d_cube(
            screen,
            center=cube["pos"],
            size=cube["size"],
            depth=cube["depth"],
            filled=True
        )

    # active cube
    draw_3d_cube(
        screen,
        center=active_pos,
        size=cube_size,
        depth=depth,
        filled=False
    )

    font = pygame.font.SysFont(None, 32)
    status = "RIGHT FIST" if is_right_fist else "RIGHT OPEN"
    screen.blit(font.render(status, True, WHITE), (20, 20))

    pygame.display.flip()
    clock.tick(30)
