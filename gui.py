import PIL.Image
from tkinter import *
from PIL import ImageTk

class ImageFrame(Frame):
    def __init__(self,master, *args, **kwargs):
        Frame.__init__(self,master=master, *args, **kwargs)
        self.x = self.y = 0
        self.canvas = Canvas(self, cursor="cross", **kwargs)

        self.sbarv=Scrollbar(self,orient=VERTICAL)
        self.sbarh=Scrollbar(self,orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        self.sbarv.grid(row=0,column=1,stick=N+S)
        self.sbarh.grid(row=1,column=0,sticky=E+W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self.im = PIL.Image.open("../Data/frames/frame_0.jpg")
        self.wazil,self.lard=self.im.size
        
        self.canvas.config(scrollregion=(0,0,self.wazil,self.lard))
        self.tk_im = ImageTk.PhotoImage(self.im)
        
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)   


    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # create rectangle if not yet exist
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)

        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if event.x > 0.9*w:
            self.canvas.xview_scroll(1, 'units') 
        elif event.x < 0.1*w:
            self.canvas.xview_scroll(-1, 'units')
        if event.y > 0.9*h:
            self.canvas.yview_scroll(1, 'units') 
        elif event.y < 0.1*h:
            self.canvas.yview_scroll(-1, 'units')

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)    

    def on_button_release(self, event):
        self.end_x = self.canvas.canvasx(event.x)
        self.end_y = self.canvas.canvasy(event.y)

class InputFrame(Frame):
    def __init__(self,master, *args, **kwargs):
        Frame.__init__(self,master=None, *args, **kwargs)
        
        self.title = Label(self, text="Parameters:", font=("Arial 25"))
        self.title.grid(column=0, row=0, padx=10, pady=(10, 50))
        
        self.prompy_label = Label(self, text="Prompt:", font=("Arial 12"))
        self.prompy_label.grid(column=0, row=1, padx=10, pady=5)
        
        self.prompt_entry = Entry(self)
        self.prompt_entry.grid(column=0, row=2, padx=10, pady=(0, 20))
        
        self.type_of_object_label = Label(self, text="Type of object:", font=("Arial 12"))
        self.type_of_object_label.grid(column=0, row=3, padx=10, pady=5)
        
        self.type_of_object = StringVar(self, '1')
        
        Radiobutton(self, text = "Static", variable=self.type_of_object, value='1').grid(column=0, row=4, padx=10, pady=2)
        Radiobutton(self, text = "Dynamic", variable=self.type_of_object, value='2').grid(column=0, row=5, padx=10, pady=2)
        
        self.start_inpaint = Button(self, text="Start Inpaint!")
        self.start_inpaint.grid(column=0, row=6, padx=10, pady=(100, 10))

def start_inpainting_button_pressed():
    prompt = input_frame.prompt_entry.get()
    mode = input_frame.type_of_object.get()
    rect = (app.start_x, app.start_y, app.end_x, app.end_y)
        
if __name__ == "__main__":
    global app, input_frame
    
    root=Tk()
    root.title("Stable Diffusion inpainting with ControlNet")
    root.columnconfigure(2)
    root.geometry("1600x740")
    root.resizable(0, 0)
    
    app = ImageFrame(root, width=1280, height=720)
    app.pack(expand= True, fill = BOTH, side = LEFT)
    
    input_frame = InputFrame(root)
    input_frame.pack(expand= True, fill = BOTH, side = LEFT)
    
    root.mainloop()