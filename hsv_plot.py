import numpy as np
import tkinter as tk   
from PIL import Image, ImageTk
import colorsys
import matplotlib.pyplot as plt
import sys

# rotation matrix
def rotate(x, theta):
    r = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return x @ r

# update plot colors
def update(fig, line, c):
    for i in range(len(line)):
        line[i].set_color(c[i])

    fig.canvas.draw()
    fig.canvas.flush_events()

# return index of closest pair
def closest_index(data, event):
    diff = np.abs(data - event)
    return np.argmin(np.sum(diff, axis=1), axis=0)

# return r s.t. ||v += r|| = c
def normalize(v, c):
    norm = np.linalg.norm(v)
    return (norm - c) * (-v / norm)

# main GUI class
class HSV(tk.Frame):
    def __init__(self, fig, line):
        # organize windows
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0,0,750,500)

        root = tk.Tk()
        root.title('hsvplot')
        root.geometry('250x300+825+0')
        self.fig= fig
        self.line = line
        self.update = update
 
        # create a canvas
        self.dim = 250
        self.c = self.dim/2 - 0.5
        self.canvas = tk.Canvas(width=self.dim, height=self.dim)
        self.canvas.pack(fill="both", expand=True)

        # hsv image
        img = np.uint8(255*self.make_wheel(1, first=True)) 
        img = ImageTk.PhotoImage(image=Image.fromarray(img)) 
        self.hsv_cont = self.canvas.create_image(self.dim/2, self.dim/2, image=img)
                
        # create n points on canvas
        self.n = len(line)
        self.index = 0 # the current data point that is dragged

        point = [75,0]
        for i in range(self.n):
            self.create_token(point[0], point[1])
            point = rotate(point, 2*np.pi/self.n)
        
        # add bindings for clicking, dragging and releasing over
        self.canvas.tag_bind("token", "<ButtonPress-1>", self.drag_start)
        self.canvas.tag_bind("token", "<B1-Motion>", self.drag)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self.drag_end)

        # button
        self.button_quit = tk.Button(master=root, text="Quit", command=root.quit)
        self.toggle_group = tk.Button(master=root, text="Groups", relief="raised", 
                                      command=self.toggle)
        self.group = False

        # slider for v in hsv
        self.v = 1 
        self.slider = tk.Scale(root, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                              command=self.make_wheel, label="Value (HSV)", 
                              resolution=0.01)
        self.slider.set(1)
 
        # packing
        self.button_quit.pack(side=tk.LEFT, anchor='se')
        self.toggle_group.pack(side=tk.LEFT, anchor='se')
        self.slider.pack(side=tk.RIGHT, anchor='sw')

        # why twice???
        self.plot()
        self.plot()

        root.mainloop()
    
    # toggle button
    def toggle(self):
        if self.toggle_group.config('relief')[-1] == 'sunken':
            self.toggle_group.config(relief="raised")
            self.group = False
        else:
            self.toggle_group.config(relief="sunken")
            self.group = True
        print(self.group)

    # create circles for points on wheel
    def create_token(self, x, y, w=5):
        self.canvas.create_oval(
            x - w + self.dim/2,
            y - w + self.dim/2,
            x + w + self.dim/2,
            y + w + self.dim/2,
            outline="black",
            fill="black",
            tags=("token",),
        )
    
    # get coordinates of dots
    def dot_coords(self, w=5):
        dots = np.empty((self.n, 2))
        for i in range(self.n):    
            oval = self.canvas.coords(i + 2)
            dots[i] = [oval[0] + w, oval[1] + w]
        return dots

    # begin drag motion for circles
    def drag_start(self, event):
        # record the item and its location
        self.index = closest_index(self.dot_coords(), [event.x, event.y])

    # reset dist vectors if mouse outside hsv wheel
    def drag_end(self, event):
        return

    # move points in circle
    def drag(self, event):
        ind = self.index

        # mouse must be inside window
        if event.x < 0 or event.y < 0 or event.x > self.dim-1 or event.y > self.dim-1:
            return
        
        coords = self.dot_coords()
        delta = np.array([event.x - coords[self.index, 0],
                          event.y - coords[self.index, 1]])

        # move each object (radial + angular sym)
        for i in range(self.n):
            # move point 
            #self.data[self.index] += delta # point location tracking
            #if np.linalg.norm(self.data[self.index]) >= self.c: # check for wheel bound
            #    u_data = self.data[self.index] / np.linalg.norm(self.data[self.index])
            #    tan = delta - (delta @ u_data) * u_data
            #    self.canvas.move(self.index+2, tan[0], tan[1])
            #else:

            self.canvas.move(self.index+2, delta[0], delta[1])
            
            # rotate the change vectors for each point
            delta = rotate(delta, 2*np.pi/self.n)
            
            # update index
            self.index += 1 
            if self.index >= self.n:
                self.index = 0

        self.index = ind # reset index
        self.plot()

    # make hsv wheel, called upon changing value
    def make_wheel(self, v0, first=False):
        v0 = float(v0)
        self.v = v0 # for returing hsv value

        img = np.zeros((self.dim, self.dim, 3))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):        
                v = v0 # v in hsv

                s = np.sqrt((i-self.c)**2 + (j-self.c)**2) / self.c # s in hsv
                if s > 1:
                    s, v = 0, 1 # set background white

                h = np.arctan2(i-self.c, j-self.c) #(y,x), h in hsv
                if h < 0:
                    h += 2*np.pi # correct so h in [0, 2 pi]
                h /= 2*np.pi # normalize [0, 1]
                
                img[i,j] = colorsys.hsv_to_rgb(h,s,v)

        if first: # to make first image
            return img

        self.plot()

        img = np.uint8(255*img)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.hsv_cont, image=img)
        self.canvas.draw()

    # update the matplotlib figure
    def plot(self):
        rgb = np.zeros((self.n,3))

        # get h and s values
        coords = self.dot_coords() - self.c

        s = np.sqrt((coords[0,0])**2 + (coords[0,1])**2) / self.c # s in hsv
        if s > 1:
            s = 1
        h = np.arctan2(coords[0,1], coords[0,0])

        for i in range(self.n):
            if h < 0:
                h += 2*np.pi # correct so h in [0, 2 pi]

            rgb[i] = colorsys.hsv_to_rgb(h/(2*np.pi), s, self.v)
            h += (2*np.pi)/self.n

        self.update(self.fig, self.line, rgb)
