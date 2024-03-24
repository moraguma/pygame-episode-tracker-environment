import pygame

# https://github.com/openai/gym/wiki/Riverraid-v0#action

KEY_MAPPING = {(pygame.K_SPACE,): 1,
               (pygame.K_UP,): 2,
               (pygame.K_RIGHT,): 3,
               (pygame.K_LEFT,): 4,
               (pygame.K_DOWN,): 5,
               (pygame.K_UP, pygame.K_RIGHT): 6,
               (pygame.K_UP, pygame.K_LEFT): 7,
               (pygame.K_DOWN, pygame.K_RIGHT): 8,
               (pygame.K_DOWN, pygame.K_LEFT): 9,
               (pygame.K_UP, pygame.K_SPACE): 10,
               (pygame.K_RIGHT, pygame.K_SPACE): 11,
               (pygame.K_LEFT, pygame.K_SPACE): 12,
               (pygame.K_DOWN, pygame.K_SPACE): 13,
               (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): 14,
               (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): 15,
               (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): 13,
               (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): 14,}