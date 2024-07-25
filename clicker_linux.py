import tkinter as tk
import customtkinter as ctk
import pyautogui
import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import subprocess
import mss

model = YOLO('best.pt')

def click_at_point(x, y):
    print(f"Clicking at ({x}, {y})")  # برای دیباگ
    subprocess.run(["xdotool", "mousemove", str(x), str(y)])
    subprocess.run(["xdotool", "click", "1"])

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

x1, y1, x2, y2 = 0, 0, 0, 0
show_mouse_coords = False

def enter_coordinates():
    global x1, y1, x2, y2
    
    def save_coordinates():
        global x1, y1, x2, y2
        x1 = int(entry_x1.get())
        y1 = int(entry_y1.get())
        x2 = int(entry_x2.get())
        y2 = int(entry_y2.get())
        coord_window.destroy()

    coord_window = tk.Tk()
    coord_window.title("Enter Coordinates")

    tk.Label(coord_window, text="x1:").grid(row=0)
    tk.Label(coord_window, text="y1:").grid(row=1)
    tk.Label(coord_window, text="x2:").grid(row=2)
    tk.Label(coord_window, text="y2:").grid(row=3)

    entry_x1 = tk.Entry(coord_window)
    entry_y1 = tk.Entry(coord_window)
    entry_x2 = tk.Entry(coord_window)
    entry_y2 = tk.Entry(coord_window)

    entry_x1.grid(row=0, column=1)
    entry_y1.grid(row=1, column=1)
    entry_x2.grid(row=2, column=1)
    entry_y2.grid(row=3, column=1)

    tk.Button(coord_window, text='Save', command=save_coordinates).grid(row=4, column=1, pady=4)

    coord_window.mainloop()

    print(f"Selected area: ({x1}, {y1}) to ({x2}, {y2})")

def start_scanning():
    duration = int(duration_entry.get())
    end_time = time.time() + duration
    threads = []
    x1=1400
    x2=1910
    y1=0
    y2=800
    sct = mss.mss()
    
    while time.time() < end_time:
        monitor = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        results = model(screenshot, conf=float(conf_entry.get()))
        try:
            fallout = 0
            result = results[0].boxes.xyxy.cpu().numpy()  # انتقال Tensor به CPU و تبدیل به numpy array
            result = result[np.argsort(result[:, 3]*-1)]
            for box in result[:1]:
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2) + fallout
                
                thread = threading.Thread(target=click_at_point, args=[x_center + x1, y_center + y1])
                threads.append(thread)
                thread.start()                
                fallout += 22
        except Exception as e:
            print("Error:", e)
    
    for thread in threads:
        thread.join()

def toggle_mouse_coords():
    global show_mouse_coords
    show_mouse_coords = not show_mouse_coords
    if show_mouse_coords:
        update_mouse_coords()

def update_mouse_coords():
    if show_mouse_coords:
        x, y = pyautogui.position()
        mouse_coords_entry.delete(0, tk.END)
        mouse_coords_entry.insert(0, f"x: {x}, y: {y}")
        root.after(100, update_mouse_coords)  

root = ctk.CTk()
root.title("YOLO Object Detector")

frame = ctk.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = ctk.CTkLabel(master=frame, text="YOLO Object Detector")
label.pack(pady=12, padx=10)

duration_label = ctk.CTkLabel(master=frame, text="Enter duration (seconds):")
duration_label.pack(pady=12, padx=10)

duration_entry = ctk.CTkEntry(master=frame)
duration_entry.pack(pady=12, padx=10)
duration_entry.insert(0, "35")

select_area_button = ctk.CTkButton(master=frame, text="Enter Coordinates", command=enter_coordinates)
select_area_button.pack(pady=12, padx=10)

mouse_coords_button = ctk.CTkButton(master=frame, text="Toggle Mouse Coords", command=toggle_mouse_coords)
mouse_coords_button.pack(pady=12, padx=10)

mouse_coords_entry = ctk.CTkEntry(master=frame)
mouse_coords_entry.pack(pady=12, padx=10)

duration_label = ctk.CTkLabel(master=frame, text="Enter confidence factor (between 0 and 1):")
duration_label.pack(pady=12, padx=10)

conf_entry = ctk.CTkEntry(master=frame)
conf_entry.pack(pady=12, padx=10)
conf_entry.insert(0, "0.85")

start_button = ctk.CTkButton(master=frame, text="Start", command=start_scanning)
start_button.pack(pady=12, padx=10)

def close_application():
    root.quit()
    
close_button = ctk.CTkButton(master=frame, text="Close", command=close_application)
close_button.pack(side=tk.BOTTOM, pady=12, padx=10)

root.mainloop()
