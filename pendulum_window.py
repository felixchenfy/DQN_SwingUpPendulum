
# This script creates a canvas where there is a single pendulum
# (Though I created 2 links, 1 ball, and several walls, I didn't show them all in canvas.)

# The user could press keyboard "a" and "d" to change the angle of pendulum
# (For dynamics simualtion, please refer to "pendulum_simulation.py")

import tkinter as tk
import numpy as np
import math
import os
import time

class myWindow(object):
    def __init__(self):
        self.init_window()
        self.reset()
        return

    # Set Window size and add Canvas to it
    def init_window(self):

        # set window size
        WINDOW_ROWS = 600
        WINDOW_COLS = 400

        # open a window and canvas
        self.window = tk.Tk()
        self.window.title('Feiyu Chen - Inverted or Swing up Pendulum Game')
        self.window.geometry(
            '{0}x{1}+500+150'.format(WINDOW_COLS, WINDOW_ROWS))
        self.canvas = tk.Canvas(self.window, bg='white',
                                height=WINDOW_ROWS, width=WINDOW_COLS)

        # save window configs
        self.WINDOW_ROWS = WINDOW_ROWS
        self.WINDOW_COLS = WINDOW_COLS

    # Reset canvas to initial state
    def reset(self):
        self.reset_canvas()
        self.bind_keys()
        self.display_text()

    # When user presses keys, call self.event_KeyPress/Release
    def bind_keys(self):
        self.canvas.bind("<KeyPress>", self.event_KeyPress)
        self.canvas.bind("<KeyRelease>", self.event_KeyRelease)
        self.canvas.pack()
        self.canvas.focus_set()

    # Display text (label) on the canvas, to tell user about the game rule
    def display_text(self):
        str_text = "User input:\n"
        str_text += "A,D: rotate link1\n"
        str_text += "W,S: rotate link2\n"
        str_text += "Q: reset game\n"
        self.display_text_(str_text)

    # Inner function for text display settings
    def display_text_(self, str_text):
        fontsize = 15
        Nlines = str_text.count("\n")
        xpos = 30
        ypos = 30+Nlines*fontsize+(Nlines-1)*10
        self.canvas.create_text(xpos, ypos, text=str_text, font=(
            "Purisa", fontsize), anchor=tk.SW)

        # A template of addling label
        # l = tk.Label(mw.window,
        #     text='OMG! this is TK!',    # text of the label
        #     bg='green',     # background color
        #     font=('Arial', 12),     # font and font size
        #     width=15, height=2  # length, width of the lable
        #     )
        # l.pack()    # put it onto canvas
        # self.canvas.create_window(100, 100, window=l)

    # Set/Reset canvas's objects: links, walls, ball, etc.s
    def reset_canvas(self, link1_angle_=0, link2_angle_=0):
        self.canvas.delete("all")

        # coordinate description:
        #       origin at left up corner.
        # x points to the right
        # y points down.

        # Notes about geometry:
        # The link has no width, its just (p0, p1)!
        # The ball is a mass point (x0, y0)!

        # ---------------------------------
        # configurations of size

        BASE_LINK_X = 200
        BASE_LINK_Y = 300
        BASE_LINK_RADIUS = 10
        BASE_LINK_COLOR = 'red'

        LINK1_LENGTH = 150
        LINK1_WIDTH = 10
        LINK1_COLOR = 'blue'

        LINK2_LENGTH = 10
        LINK2_WIDTH = 10
        LINK2_COLOR = 'green'

        BALL_COLOR = 'white'
        BALL_RADIUS = 10

        WALL_COLOR = 'black'
        WALL_LEFT_PADDING = 1
        WALL_RIGHT_PADDING = 1
        WALL_DOWN_PADDING = 1


        # ---------------------------------
        # set variables
        link1_angle = link1_angle_
        link2_angle = link2_angle_
        ball_x = 9999
        ball_y = 9999

        # ---------------------------------
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

        # Add two light
        self.add_two_lights()

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

    # Add two lights at the bottom of window, for displaying user input.
    # (you need to manually set it when key press is detected)
    def add_two_lights(self):
        LIGHT_RADIUS = 10
        LIGHT_MID = self.WINDOW_COLS/2
        LIGHT_OFFSET = self.WINDOW_COLS/4
        LIGHT_Y=500

        x0 = LIGHT_MID-LIGHT_OFFSET-LIGHT_RADIUS
        y0 = LIGHT_Y-LIGHT_RADIUS
        x1 = x0+LIGHT_RADIUS*2
        y1 = y0+LIGHT_RADIUS*2
        self.light1 = self.canvas.create_oval(x0, y0, x1, y1, fill="black")

        x0 = LIGHT_MID+LIGHT_OFFSET-LIGHT_RADIUS
        y0 = LIGHT_Y-LIGHT_RADIUS
        x1 = x0+LIGHT_RADIUS*2
        y1 = y0+LIGHT_RADIUS*2
        self.light2 = self.canvas.create_oval(x0, y0, x1, y1, fill="black")

        self.canvas.create_text(LIGHT_MID, LIGHT_Y, text="    User Input\n<--  Indicator -->", font=(
            "Purisa", 15))

    # get coordinates of an object (ball, link, wall, etc)
    def get_coords(self, obj):
        coords = self.canvas.coords(obj)
        x0 = coords[0]
        y0 = coords[1]
        x1 = coords[2]
        y1 = coords[3]
        theta = np.arctan2(y1-y0, x1-x0)
        return x0, y0, x1, y1, theta

    # move ball with a distance of dx and dy
    def move_ball(self, dx, dy, if_update=True):
        self.ball_x += dx
        self.ball_y += dy
        if if_update:
            self.update_ball()

    def move_ball_to(self, x, y, if_update=True):
        self.move_ball(dx=x-self.ball_x, dy=y-self.ball_y)
        self.ball_x = x
        self.ball_y = y

    # rotate link1 with an angle of dtheta
    def rotate_link1(self, dtheta, if_update=True):
        self.link1_theta += dtheta
        if if_update:
            self.update_link1()
            self.update_link2()  # since link1 changes, link2 also changes

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

    # After changing link1_theta, call this function.
    # Feed current link1's angle into Canvas to update it.
    # (To immediately display the new image, you might need to call self.render())
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
        # None
        return

    def event_KeyPress(self, e):
        c = e.char.lower() # to make, e.g. "X" and "x" the same key press
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

        # reset
        elif c == 'q':
            self.reset()

        return

    def mainloop(self):
        self.canvas.mainloop()

    def after(self, time, func):
        self.window.after(time, func)

    def render(self):
        self.canvas.update()

if __name__ == "__main__":
    os.system('xset r off') # turn off continuous keypress
    
    mw = myWindow()

    tk.mainloop()

    os.system('xset r on') # turn off continuous keypress
