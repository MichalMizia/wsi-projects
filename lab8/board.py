import pygame
import numpy as np
import sys
from MnistLoader import MnistDataloader
from Perceptron import Perceptron
from scipy.ndimage import gaussian_filter


class Settings:
    SCREEN_WIDTH = 560
    SCREEN_HEIGHT = 560
    CELL_SIZE = 20
    WIDTH_CELL_NUMBER = SCREEN_WIDTH // CELL_SIZE
    HEIGHT_CELL_NUMBER = SCREEN_HEIGHT // CELL_SIZE
    BACKGROUND_COLOR = (255, 255, 255)
    DEFAULT_COLOR = (255, 255, 255)
    PAINT_COLOR = (0, 0, 0)
    LIGHT_GRAY = (150, 150, 150)
    BUTTON_COLOR = (0, 0, 255)
    BUTTON_HOVER_COLOR = (0, 0, 200)
    BUTTON_TEXT_COLOR = (255, 255, 255)
    BUTTON_OUTLINE_COLOR = (0, 0, 0)
    BORDER_COLOR = (0, 0, 0)
    MARGIN = 16
    PADDING = 20


class Cell:
    def __init__(self, dimension, x_coord, y_coord):
        self.dimension = dimension
        self.x = x_coord
        self.y = y_coord

        self.rect = pygame.Rect(self.x, self.y, self.dimension, self.dimension)
        self.color = Settings.DEFAULT_COLOR

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

    def set_color(self, color):
        self.color = color


class Board:
    def __init__(self, screen, x_offset=0, y_offset=0):
        self.board_width = Settings.SCREEN_WIDTH
        self.board_height = Settings.SCREEN_HEIGHT
        self.cell_size = Settings.CELL_SIZE
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.screen = screen
        self.is_painting = False
        self.cells = []
        self.reset_board()

    def reset_board(self):
        self.cells = []
        for x in range(0, self.board_width, self.cell_size):
            cell_row = []
            for y in range(0, self.board_height, self.cell_size):
                cell = Cell(self.cell_size, x + self.x_offset, y + self.y_offset)
                cell_row.append(cell)

            self.cells.append(cell_row)

    def set_draw_mode(self):
        self.is_painting = True

    def unset_draw_mode(self):
        self.is_painting = False

    def draw_board(self):
        for cell_row in self.cells:
            for cell in cell_row:
                cell.draw(self.screen)
        pygame.draw.rect(
            self.screen,
            Settings.PAINT_COLOR,
            pygame.Rect(
                self.x_offset, self.y_offset, self.board_width, self.board_height
            ),
            2,
        )

    def draw_cell(self, x, y, color=Settings.PAINT_COLOR):
        if (
            x < 0
            or x >= Settings.WIDTH_CELL_NUMBER
            or y < 0
            or y >= Settings.HEIGHT_CELL_NUMBER
        ):
            return
        cell = self.cells[x][y]
        cell.set_color(color)
        cell.draw(self.screen)

    def restart(self):
        self.reset_board()
        self.draw_board()

    def get_image_as_np_array(self):
        matrix = []
        for y in range(Settings.HEIGHT_CELL_NUMBER):
            row = []
            for x in range(Settings.WIDTH_CELL_NUMBER):
                cell = self.cells[x][y]
                if cell.color == Settings.DEFAULT_COLOR:
                    row.append(0)
                elif cell.color == Settings.PAINT_COLOR:
                    row.append(1)
                elif cell.color == Settings.LIGHT_GRAY:
                    row.append(1)
            matrix.append(row)

        np_array = np.array(matrix)
        np_array = gaussian_filter(np_array, sigma=0.5, radius=3)

        np_array = np.array(matrix).reshape((784, 1))
        return np_array


class Button:
    def __init__(self, x, y, width, height, text, filled=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.filled = filled
        self.color = Settings.BUTTON_COLOR
        self.hover_color = Settings.BUTTON_HOVER_COLOR
        self.text_color = Settings.BUTTON_TEXT_COLOR
        self.outline_color = Settings.BUTTON_OUTLINE_COLOR

    def draw(self, screen, font):
        if self.filled:
            pygame.draw.rect(screen, self.color, self.rect)
        else:
            pygame.draw.rect(screen, self.outline_color, self.rect, 2)

        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_hovered(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


class DigitPainter:
    def __init__(self):
        mnist_dataloader = MnistDataloader()

        (x_train, y_train), (x_test, y_test) = (
            mnist_dataloader.load_data()
        )  # as np arrays

        x_train = x_train.T
        y_train = y_train.T

        network = Perceptron(784, hidden_layers=[50, 50], output_layer=10, lr=0.1)
        network.train(
            x_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(x_test.T, y_test.T),
        )

        self.model = network

        print("--------------------------------------------")
        print("              Network trained               ")
        print("--------------------------------------------")

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                Settings.SCREEN_WIDTH + 360 + 2 * Settings.PADDING,
                Settings.SCREEN_HEIGHT + 2 * Settings.PADDING,
            )
        )
        self.screen.fill(Settings.BACKGROUND_COLOR)
        pygame.display.set_caption("DigitPainter")

        self.board = Board(
            self.screen, x_offset=Settings.PADDING + 10, y_offset=Settings.PADDING + 10
        )
        self.font = pygame.font.SysFont("Arial", 36)
        self.prediction = None

        self.predict_button = Button(
            Settings.SCREEN_WIDTH + Settings.PADDING + 30,
            Settings.PADDING + 10,
            150,
            50,
            "Predict",
            filled=True,
        )
        self.predict_button.color = (75, 0, 130)  # type: ignore
        self.predict_button.text_color = (255, 255, 255)  # type: ignore

        self.clear_button = Button(
            Settings.SCREEN_WIDTH + Settings.PADDING + 200,
            Settings.PADDING + 10,
            150,
            50,
            "Clear",
            filled=False,
        )
        self.clear_button.outline_color = (75, 0, 130)  # type: ignore
        self.clear_button.text_color = (75, 0, 130)  # type: ignore

    def run(self):
        while True:
            self.check_events()
            self.update_screen()
            pygame.display.flip()

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.check_mouse_button_down_events(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.check_mouse_button_up_events(event)
            elif event.type == pygame.KEYDOWN:
                self.check_keydown_events(event)

    def check_mouse_button_down_events(self, event):
        if event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            if self.predict_button.is_hovered(mouse_pos):
                self.make_prediction()
            elif self.clear_button.is_hovered(mouse_pos):
                self.board.restart()
                self.prediction = None
            else:
                self.board.set_draw_mode()

    def check_mouse_button_up_events(self, event):
        if event.button == 1:
            self.board.unset_draw_mode()

    def check_keydown_events(self, event):
        if event.key == pygame.K_q:
            sys.exit()
        elif event.key == pygame.K_c:
            self.board.restart()
            self.prediction = None
        elif event.key == pygame.K_SPACE:
            self.make_prediction()

    def make_prediction(self):
        np_array = self.board.get_image_as_np_array()
        self.prediction = np.argmax(self.model.forward_prop(np_array))

    def update_screen(self):
        self.screen.fill(Settings.BACKGROUND_COLOR)
        self.board.draw_board()
        self.predict_button.draw(self.screen, self.font)
        self.clear_button.draw(self.screen, self.font)

        if self.prediction is not None:
            prediction_text = self.font.render(
                f"Prediction: {self.prediction}", True, (0, 0, 0)
            )
            self.screen.blit(
                prediction_text, (Settings.SCREEN_WIDTH + Settings.PADDING + 30, 76)
            )

        lorem_text = "Try to draw the digits in the center as the mnist dataset contains mostly digits that do not occupy the full 28x28, they are closer to 18x18. "
        wrapped_text = self.wrap_text(lorem_text, self.font, 360)
        y_offset = Settings.SCREEN_HEIGHT + Settings.PADDING - len(wrapped_text) * 36
        for line in wrapped_text:
            text_surface = self.font.render(line, True, (0, 0, 0))
            self.screen.blit(
                text_surface, (Settings.SCREEN_WIDTH + Settings.PADDING + 20, y_offset)
            )
            y_offset += 36

        if self.board.is_painting:
            x, y = pygame.mouse.get_pos()
            x, y = (x - Settings.PADDING - 10) // Settings.CELL_SIZE, (
                y - Settings.PADDING - 10
            ) // Settings.CELL_SIZE
            if (
                x < 0
                or x >= Settings.WIDTH_CELL_NUMBER
                or y < 0
                or y >= Settings.HEIGHT_CELL_NUMBER
            ):
                return
            self.board.draw_cell(x, y)

            if x - 1 >= 0 and self.board.cells[x - 1][y].color[0] == 255:
                self.board.draw_cell(x - 1, y, Settings.LIGHT_GRAY)
            if (
                x + 1 < Settings.WIDTH_CELL_NUMBER
                and self.board.cells[x + 1][y].color[0] == 255
            ):
                self.board.draw_cell(x + 1, y, Settings.LIGHT_GRAY)
            if y - 1 >= 0 and self.board.cells[x][y - 1].color[0] == 255:
                self.board.draw_cell(x, y - 1, Settings.LIGHT_GRAY)
            if (
                y + 1 < Settings.HEIGHT_CELL_NUMBER
                and self.board.cells[x][y + 1].color[0] == 255
            ):
                self.board.draw_cell(x, y + 1, Settings.LIGHT_GRAY)

    def wrap_text(self, text, font, max_width):
        words = text.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        return lines


def main():
    painter = DigitPainter()
    painter.run()


if __name__ == "__main__":
    main()
