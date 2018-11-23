
# This script creates a canvas, and dispalys the pendulum
# For dynamics and simualtion, please refer to "pendulum_simulation.py"

import tkinter as tk
import numpy as np
import math
import os
import time


class myWindow(object):
    def __init__(self):
        self.init_window()
        self.restart()
        self.bind_keys()
        self.display_text()
        return

    def init_window(self):

        # set window size
        WINDOW_ROWS = 600
        WINDOW_COLS = 400

        # open a window and canvas
        self.window = tk.Tk()
        self.window.title('my window')
        self.window.geometry(
            '{0}x{1}+500+150'.format(WINDOW_COLS, WINDOW_ROWS))
        self.canvas = tk.Canvas(self.window, bg='white',
                                height=WINDOW_ROWS, width=WINDOW_COLS)

        # save window configs
        self.WINDOW_ROWS = WINDOW_ROWS
        self.WINDOW_COLS = WINDOW_COLS

    def bind_keys(self):
        self.canvas.bind("<KeyPress>", self.event_KeyPress)
        self.canvas.bind("<KeyRelease>", self.event_KeyRelease)
        self.canvas.pack()
        self.canvas.focus_set()

    def display_text(self):

        str_text = "User input:\n"
        str_text += "A,D: rotate link1\n"
        str_text += "W,S: rotate link2\n"
        # str_text += "J,K: apply torque to link1\n"
        str_text += "Q: reset game\n"
        # str_text+="E/R: enable/disable simulation\n" # the sim engine is in "pendulum_env"
        self.display_text_(str_text)

    def display_text_(self, str_text):
        fontsize = 15
        Nlines = str_text.count("\n")
        xpos = 30
        ypos = 30+Nlines*fontsize+(Nlines-1)*10
        self.canvas.create_text(xpos, ypos, text=str_text, font=(
            "Purisa", fontsize), anchor=tk.SW)

        # Add label
        # l = tk.Label(mw.window,
        #     text='OMG! this is TK!',    # 标签的文字
        #     bg='green',     # 背景颜色
        #     font=('Arial', 12),     # 字体和字体大小
        #     width=15, height=2  # 标签长宽
        #     )
        # l.pack()    # 固定窗口位置
        # self.canvas.create_window(100, 100, window=l)

    def restart(self):
        self.canvas.delete("all")

        # coordinate
        # coordinate: origin at left up corner.
        # x points to the right
        # y points down.

        # Important consumptions:
        # The link has no width!
        # The ball is a mass point !

        # configurations of size

        BASE_LINK_X = 200
        BASE_LINK_Y = 400
        BASE_LINK_RADIUS = 10
        BASE_LINK_COLOR = 'black'

        LINK1_LENGTH = 150
        LINK1_WIDTH = 10
        LINK1_COLOR = 'blue'

        LINK2_LENGTH = 10
        LINK2_WIDTH = 10
        LINK2_COLOR = 'red'

        ball_x = 9999
        ball_y = 9999
        BALL_COLOR = 'white'
        BALL_RADIUS = 10

        WALL_COLOR = 'black'
        WALL_LEFT_PADDING = 1
        WALL_RIGHT_PADDING = 1
        WALL_DOWN_PADDING = 1

        # configurations of dynamics: mass, inertia, force, friction, etc.

        # set variables
        link1_angle = 0
        link2_angle = 0

        # draw all staff
        # 1. base link
        x0 = BASE_LINK_X-BASE_LINK_RADIUS
        y0 = BASE_LINK_Y-BASE_LINK_RADIUS
        x1 = BASE_LINK_X+BASE_LINK_RADIUS
        y1 = BASE_LINK_Y+BASE_LINK_RADIUS
        base_link = self.canvas.create_oval(
            x0, y0, x1, y1, fill=BASE_LINK_COLOR)

        # 2. link1
        link1_end_x = BASE_LINK_X+LINK1_LENGTH*math.cos(link1_angle)
        link1_end_y = BASE_LINK_Y+LINK1_LENGTH*math.sin(link1_angle)
        x0 = BASE_LINK_X
        y0 = BASE_LINK_Y
        x1 = link1_end_x
        y1 = link1_end_y
        link1 = self.canvas.create_line(
            x0, y0, x1, y1, width=LINK1_WIDTH, fill=LINK1_COLOR)

        # 3. link2
        link2_end_x = link1_end_x+LINK2_LENGTH * \
            math.cos(link1_angle+link2_angle)
        link2_end_y = link1_end_y+LINK2_LENGTH * \
            math.sin(link1_angle+link2_angle)
        x0 = link1_end_x
        y0 = link1_end_y
        x1 = link2_end_x
        y1 = link2_end_y
        link2 = self.canvas.create_line(
            x0, y0, x1, y1, width=LINK2_WIDTH, fill=LINK2_COLOR)

        # 4. ball
        x0 = ball_x-BALL_RADIUS
        y0 = ball_y-BALL_RADIUS
        x1 = ball_x+BALL_RADIUS
        y1 = ball_y+BALL_RADIUS
        ball = self.canvas.create_oval(x0, y0, x1, y1, fill=BALL_COLOR)

        # 5. walls-right
        x0 = self.WINDOW_COLS-WALL_RIGHT_PADDING
        x1 = self.WINDOW_COLS
        y0 = 0
        y1 = self.WINDOW_ROWS
        wall_right = self.canvas.create_rectangle(
            x0, y0, x1, y1, fill=WALL_COLOR)

        # 5. walls-down
        x0 = 0
        x1 = self.WINDOW_COLS
        y0 = self.WINDOW_ROWS-WALL_DOWN_PADDING
        y1 = self.WINDOW_ROWS
        wall_down = self.canvas.create_rectangle(
            x0, y0, x1, y1, fill=WALL_COLOR)

        # 5. walls-left
        x0 = 0
        x1 = WALL_LEFT_PADDING
        y0 = 0
        y1 = self.WINDOW_ROWS
        wall_left = self.canvas.create_rectangle(
            x0, y0, x1, y1, fill=WALL_COLOR)

        # put on staff onto canvas
        self.canvas.pack()

        # store to the class
        self.BASE_LINK_X = BASE_LINK_X
        self.BASE_LINK_Y = BASE_LINK_Y

        self.link1 = link1
        self.link1_length = LINK1_LENGTH
        self.link1_theta = 0

        self.link2 = link2
        self.link2_length = LINK2_LENGTH
        self.link2_theta = 0

        self.ball = ball
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.BALL_RADIUS = BALL_RADIUS

    def get_coords(self, obj):
        coords = self.canvas.coords(obj)
        x0 = coords[0]
        y0 = coords[1]
        x1 = coords[2]
        y1 = coords[3]
        theta = np.arctan2(y1-y0, x1-x0)
        return x0, y0, x1, y1, theta

    def move_ball(self, dx, dy, if_update=True):
        self.ball_x += dx
        self.ball_y += dy
        if if_update:
            self.update_ball()
        return

    def move_ball_to(self, x, y, if_update=True):
        self.move_ball(dx=x-self.ball_x, dy=y-self.ball_y)
        self.ball_x = x
        self.ball_y = y

    def rotate_link1(self, dtheta, if_update=True):
        self.link1_theta += dtheta
        if if_update:
            self.update_link1()
            self.update_link2()  # since link1 changes, link2 also changes
        return

    def rotate_link1_to(self, theta):
        self.rotate_link1(dtheta=theta-self.link1_theta)
        self.link1_theta = theta

    def rotate_link2(self, dtheta, if_update=True):
        self.link2_theta += dtheta
        if if_update:
            self.update_link2()
        return

    def rotate_link2_to(self, theta):
        self.rotate_link2(dtheta=theta-self.link2_theta)
        self.link2_theta = theta

    def update_link1(self):  # after changing link1_theta, call this function
        x0 = self.BASE_LINK_X
        y0 = self.BASE_LINK_Y
        x1 = x0 + self.link1_length*math.cos(self.link1_theta)
        y1 = y0 + self.link1_length*math.sin(self.link1_theta)
        newpose = [x0, y0, x1, y1]
        self.canvas.coords(self.link1, *newpose)
        return

    def update_link2(self):  # after changing link2_theta, call this function
        _, _, x0, y0, _ = self.get_coords(self.link1)
        theta = self.link1_theta + self.link2_theta
        x1 = x0 + self.link2_length*math.cos(theta)
        y1 = y0 + self.link2_length*math.sin(theta)
        newpose = [x0, y0, x1, y1]
        self.canvas.coords(self.link2, *newpose)
        return

    def update_ball(self):
        x0 = self.ball_x
        y0 = self.ball_y
        BALL_RADIUS = self.BALL_RADIUS
        newpose = [x0-BALL_RADIUS, y0-BALL_RADIUS,
                   x0+BALL_RADIUS, y0+BALL_RADIUS]
        self.canvas.coords(self.ball, *newpose)

    def event_KeyRelease(self, e):
        return
        
    def event_KeyPress(self, e):
        c = e.char.lower()
        # print('press down: %s, ind = %d ' %(c, ord(c)))

        # rotate link
        dtheta = 0.1
        if c == "w":
            self.rotate_link2(-dtheta)
        elif c == "s":
            self.rotate_link2(dtheta)
        elif c == "a":
            self.rotate_link1(-dtheta)
        elif c == "d":
            self.rotate_link1(dtheta)

        # move ball
        dis = 3
        if c == "i":
            self.move_ball(0, -dis)
        elif c == "k":
            self.move_ball(0, +dis)
        elif c == "j":
            self.move_ball(-dis, 0)
        elif c == "l":
            self.move_ball(+dis, 0)

        # restart
        elif c == 'q':
            self.restart()

        return

    def mainloop(self):
        self.canvas.mainloop()


def test_print():
    print(time.time())
    mw.window.after(1000, test_print)


if __name__ == "__main__":
    # b = tk.Button(window, text='move', command=moveit).pack()
    os.system('xset r on')

    mw = myWindow()
    # mw.window.after(1000, test_print)
    tk.mainloop()
