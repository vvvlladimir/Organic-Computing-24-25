from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c

def gen():
    return random.randint(0, c.GRID_LEN - 1)

class GameGrid(Frame):
    def __init__(self, grid_len=4):
        Frame.__init__(self)
        c.GRID_LEN = grid_len

        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", self.key_down)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.score = 0
        self.update_grid_cells()

        # mainloop is commented out to support API control
        # self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_QUIT:
            exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done, add_score = self.commands[key](self.matrix)
            if done:
                self.score += add_score
                print(f"Score: {self.score}")
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2

    # CONTROL: Movement management
    def move(self, direction):
        """
        Moves tiles in the specified direction.

        :param direction: one of ['up', 'down', 'left', 'right']
        :return: True if the move was successful; otherwise False.
        """
        directions = {
            'up': logic.up,
            'down': logic.down,
            'left': logic.left,
            'right': logic.right
        }

        if direction not in directions:
            raise ValueError("Invalid direction. Use 'up', 'down', 'left', 'right'.")

        self.matrix, done, add_score = directions[direction](self.matrix)
        if done:
            self.score += add_score
            self.matrix = logic.add_two(self.matrix)
            self.update_grid_cells()
        return done

    # CONTROL: Reset game
    def reset(self):
        """
        Resets the game to its initial state.
        """
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.score = 0
        self.update_grid_cells()

    # OBSERVE: Game state observation
    def state(self):
        """
        Returns the current state of the game.

        :return: A dictionary with the game's state.
        """
        return {
            "matrix": self.matrix,
            "score": self.score,
            "game_over": logic.game_state(self.matrix) == 'lose',
            "game_won": logic.game_state(self.matrix) == 'win'
        }

game = GameGrid()