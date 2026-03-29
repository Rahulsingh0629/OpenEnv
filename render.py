import pygame
import sys

CELL_SIZE = 50
GRID_COLOR = (50, 50, 50)
AGENT_COLORS = [(0, 255, 0), (255, 0, 0)]
RESOURCE_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)

class Renderer:
    def __init__(self, grid_size):
        pygame.init()

        self.grid_size = grid_size
        self.width = grid_size * CELL_SIZE
        self.height = grid_size * CELL_SIZE + 60  

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Mini RTS AI - Production Mode")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

   
    def draw(self, env, episode=0, rewards=None):
        self.screen.fill((0, 0, 0))

      
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * CELL_SIZE, y * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        
        for r in env.resources:
            pygame.draw.circle(
                self.screen,
                RESOURCE_COLOR,
                (r[1]*CELL_SIZE + CELL_SIZE//2,
                 r[0]*CELL_SIZE + CELL_SIZE//2),
                10
            )

        
        for agent_id, pos in env.agent_positions.items():
            color = AGENT_COLORS[agent_id % len(AGENT_COLORS)]

            rect = pygame.Rect(
                pos[1]*CELL_SIZE,
                pos[0]*CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(self.screen, color, rect)

            
            if hasattr(env, "agent_health"):
                health = env.agent_health.get(agent_id, 100)

                pygame.draw.rect(
                    self.screen,
                    (255, 0, 0),
                    (rect.x, rect.y - 5, CELL_SIZE, 4)
                )

                pygame.draw.rect(
                    self.screen,
                    (0, 255, 0),
                    (rect.x, rect.y - 5, CELL_SIZE * (health/100), 4)
                )

        
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            (0, self.grid_size * CELL_SIZE, self.width, 60)
        )

        
        episode_text = self.font.render(f"Episode: {episode}", True, TEXT_COLOR)
        self.screen.blit(episode_text, (10, self.grid_size * CELL_SIZE + 10))

        if rewards:
            reward_text = self.font.render(f"Reward: {sum(rewards.values()):.2f}", True, TEXT_COLOR)
            self.screen.blit(reward_text, (200, self.grid_size * CELL_SIZE + 10))

        pygame.display.flip()
        self.clock.tick(30) 

    def show_winner(self, winner):
        text = "YOU WIN!" if winner == 0 else "AI WINS!"
        font = pygame.font.Font(None, 72)

        label = font.render(text, True, (255, 255, 255))
        self.screen.blit(label, (self.width//3, self.height//3)) 

        pygame.display.flip()
        pygame.time.wait(3000)

   
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    
    def get_human_action(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]: return 0
        if keys[pygame.K_DOWN]: return 1
        if keys[pygame.K_LEFT]: return 2
        if keys[pygame.K_RIGHT]: return 3

        return 4 

    
    def get_mouse_position(self):
        if pygame.mouse.get_pressed()[0]: 
            x, y = pygame.mouse.get_pos()

            grid_x = y // CELL_SIZE
            grid_y = x // CELL_SIZE

            return (grid_x, grid_y)

        return None
    
    
