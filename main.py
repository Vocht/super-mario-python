import enum
import os
import pickle
import random
from re import L
from tkinter.messagebox import NO

import neat
import pygame
from classes.Camera import Camera

from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Menu import Menu
from classes.Sound import Sound
from entities.Mario import Mario

windowSize = 640, 480

def main():
    pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    screen = pygame.display.set_mode(windowSize)
    max_frame_rate = 60
    dashboard = Dashboard("./img/font.png", 8, screen)
    sound = Sound()
    level = Level(screen, sound, dashboard)
    menu = Menu(screen, dashboard, level, sound)

    while not menu.start:
        menu.update()

    mario = Mario(0, 0, level, screen, dashboard, sound)
    clock = pygame.time.Clock()

    while not mario.restart:
        pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
        if mario.pause:
            mario.pauseObj.update()
        else:
            level.drawLevel(mario.camera)
            dashboard.update()
            mario.update()
        pygame.display.update()
        clock.tick(max_frame_rate)
    return 'restart'

def test_ai(genome):
    pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    screen = pygame.display.set_mode(windowSize)
    max_frame_rate = 60
    dashboard = Dashboard("./img/font.png", 8, screen)
    sound = Sound()
    level = Level(screen, sound, dashboard)
    menu = Menu(screen, dashboard, level, sound)

    # menu.level.loadLevel("Level1-1")
    # menu.update()
    # menu.start = True

    while not menu.start:
        menu.update()
    
    mario = Mario(0, 0, level, screen, dashboard, sound)
    clock = pygame.time.Clock()

    while not mario.restart:
        pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
        if mario.pause:
            mario.pauseObj.update()
        else:
            """
            Here we want to get the game state and feed the inputs to the neural network.
            Then we will take the outputs and feed them to the input controller of mario.
            """
            game_state = mario.game_state()
            inputs = list(game_state.values())
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            outputs = net.activate(inputs)
            decision = outputs.index(max(outputs)) # 0 = left, 1 = right, 2 = jump, 3 = nothing
            if decision == 0:
                mario.traits["goTrait"].direction = -1
            elif decision == 1:
                mario.traits["goTrait"].direction = 1
            elif decision == 2:
                mario.traits["jumpTrait"].jump(True)
            else:
                mario.traits["goTrait"].direction = 0

            level.drawLevel(mario.camera)

            draw_overlay(screen, mario, genome)

            dashboard.update()
            mario.update()

        pygame.display.update()
        clock.tick(max_frame_rate)
    # return 'restart'

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        idle_time = 0
        previous_x = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        screen = pygame.display.set_mode(windowSize)
        max_frame_rate = 600
        dashboard = Dashboard("./img/font.png", 8, screen)
        sound = Sound()
        level = Level(screen, sound, dashboard)
        menu = Menu(screen, dashboard, level, sound)

        menu.level.loadLevel("Level1-1")
        menu.update()
        menu.start = True

        while not menu.start:
            menu.update()
        
        mario = Mario(0, 0, level, screen, dashboard, sound)
        clock = pygame.time.Clock()

        while not mario.restart:
            pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
            if mario.pause:
                mario.pauseObj.update()
            else:
                """
                Here we want to get the game state and feed the inputs to the neural network.
                Then we will take the outputs and feed them to the input controller of mario.
                """
                game_state = mario.game_state()
                inputs = list(game_state.values())
                outputs = net.activate(inputs)
                decision = outputs.index(max(outputs)) # 0 = left, 1 = right, 2 = jump, 3 = nothing
                if decision == 0:
                    mario.traits["goTrait"].direction = -1
                elif decision == 1:
                    mario.traits["goTrait"].direction = 1
                elif decision == 2:
                    mario.traits["jumpTrait"].jump(True)
                else:
                    mario.traits["goTrait"].direction = 0

                level.drawLevel(mario.camera)






                # draw_overlay(screen, mario, genome)







                dashboard.update()
                mario.update()
                if mario.getPos()[0] - mario.camera.x <= previous_x:
                    idle_time += 1
                if idle_time > 600:
                    break
                previous_x = mario.getPos()[0] - mario.camera.x

            pygame.display.update()
            clock.tick(max_frame_rate)

        genome.fitness = mario.getPos()[0] - mario.camera.x
        # return 'restart'

def draw_overlay(screen, mario, genome):
    # TODO: Refactor overlay

    random.seed(50)

    overlay = pygame.Surface((240, 180))
    overlay.set_alpha(128)
    overlay.fill((255,255,255))

    game_state = mario.game_state()
    # Draw the game state to the overlay.
    for i, key in enumerate(game_state):
        if game_state[key] == 1: # Platform
            pygame.draw.rect(overlay, (0, 0, 0), (key[0] / 32 * 12, key[1] / 32 * 12, 12, 12))
        if game_state[key] == 2: # Enemies
            pygame.draw.circle(overlay, (255, 0, 0), (key[0] / 32 * 12 + 6, key[1] / 32 * 12 + 6), 6)
        if game_state[key] == 3: # Player
            pygame.draw.circle(overlay, (0, 0, 255), (key[0] / 32 * 12 + 6, key[1] / 32 * 12 + 6), 6)
    
    # Print outputs on screen
    font = pygame.font.Font('freesansbold.ttf', 16)
    left = font.render('• Left', True, (0, 0, 0))
    right = font.render('• Right', True, (0, 0, 0))
    jump = font.render('• Jump', True, (0, 0, 0))

    left_connection = (552, 78)
    right_connection = (552, 108)
    jump_connection = (552, 138)

    DARK = (0, 0, 0)
    LIGHT = (100, 100, 100)

    # Draw the neural network to the overlay.
    # Draw hidden nodes
    hidden_coordinates = {}
    for node in genome.nodes:
        if node > 3:
            node_x = random.randint(300, 500)
            node_y = random.randint(32, 113)
            pygame.draw.rect(screen, (0, 0, 0), (node_x, node_y, 7, 7), 2)
            hidden_coordinates[node] = (node_x, node_y)

    # Draw connections
    for i, key in enumerate(genome.connections):
        if key[0] < 0:
            # Connection from input node
            coordinate = list(game_state)[-key[0] - 1]
            x = coordinate[0] / 32 * 12 + 32
            y = coordinate[1] / 32 * 12 + 32
            connection = left_connection if key[1] == 0 else right_connection if key[1] == 1 else jump_connection if key[1] == 2 else None
            if connection is None and key[1] > 4:
                connection = (hidden_coordinates[key[1]][0], hidden_coordinates[key[1]][1] + 3)
            if connection is not None:
                gcon = genome.connections[key]
                pygame.draw.line(screen, DARK if gcon.enabled else LIGHT, (x + 6, y + 6), connection, 1)
        if key[0] > 0:
            # Connection from hidden node
            coordinate = hidden_coordinates[key[0]]
            connection = left_connection if key[1] == 0 else right_connection if key[1] == 1 else jump_connection if key[1] == 2 else None
            if connection is None and key[1] > 4:
                connection = (hidden_coordinates[key[1]][0], hidden_coordinates[key[1]][1] + 3)
            if connection is not None:
                gcon = genome.connections[key]
                pygame.draw.line(screen, DARK if gcon.enabled else LIGHT, (coordinate[0] + 7, coordinate[1] + 3), connection, 1)

    # Print progress percentage (max distance: 1888 pixels on x-axis)
    progress = str(int((mario.getPos()[0] - mario.camera.x) / 1888 * 100))
    progress_text = font.render('Progress: ' + progress + '%', True, (225, 225, 225))
    
    screen.blit(left, (550, 70))
    screen.blit(right, (550, 100))
    screen.blit(jump, (550, 130))
    screen.blit(overlay, (32, 32))
    progress_box = pygame.image.load('img/box.png')
    progress_box = pygame.transform.scale(progress_box, (150, 30))
    screen.blit(progress_box, (20, 440))
    screen.blit(progress_text, (32, 448))
    pygame.draw.rect(screen, (255, 255, 255), (31, 31, 241, 181), 2)

def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-65')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    
def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    test_ai(winner)

if __name__ == "__main__":
    exitmessage = 'train_ai'
    while exitmessage == 'restart':
        exitmessage = main()
    if exitmessage == 'train_ai':
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config.txt')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        run_neat(config)
        # test_best_network(config)
