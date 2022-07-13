try:
    import tkinter as tk
    from tkinter import *
    from tkinter import font as tkfont
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import tkFont as tkfont
    import Tk as root
    
import os


from Tkinter import PlayFunc
class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, IOT, ImageProcess):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Home Page", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="START",
                            command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="QUIT",
                            command=lambda: controller.show_frame(exit()))
        button1.pack()
        button2.pack()

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="WELCOME", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button1 = tk.Button(self, text="IOT",
                            command=lambda: controller.show_frame("IOT"))
        button1.pack()
        button2 = tk.Button(self, text="Image Process",
                            command=lambda: controller.show_frame("ImageProcess"))
        button2.pack()
        button3 = tk.Button(self, text="BACK",
                            command=lambda: controller.show_frame("StartPage"))
        button3.pack()

class IOT(tk.Frame):

    def __init__(self, parent, controller):
        #root.__init__loop()
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="IOT", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="BACK",
                            command=lambda: self.controller.show_frame("PageOne"))
        button1.pack()

class ImageProcess(tk.Frame):

    def __init__(self, parent, controller):
        
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Image Processing", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button3 = tk.Button(self, text="Process",
                            command=lambda:callFunc())
        button2 = tk.Button(self, text="BACK",
                            command=lambda: controller.show_frame("PageOne"))
        button3.pack()
        button2.pack()

def callFunc():
    PlayFunc()
def filedisplay(diseasename):
    
    os.startfile(diseasename)
    
    

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
