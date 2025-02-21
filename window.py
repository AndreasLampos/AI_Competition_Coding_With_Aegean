import tkinter as tk
from PIL import Image, ImageTk  # Requires `pip install pillow`

# Function to open main window
def open_main_window():
    splash.destroy()  # Close splash screen
    main_window()  # Open main window

# Function for main window
def main_window():
    root = tk.Tk()
    root.title("Aegean Demand Forecasting")
    root.geometry("1200x800")
    root.configure(bg="lightgrey")

    # Add widgets
    label = tk.Label(root, text="Enter something:", font=("Arial", 12))
    label.pack(pady=10)

    entry = tk.Entry(root, width=30)
    entry.pack(pady=5)

    button = tk.Button(root, text="Submit", command=on_button_click, bg="blue", fg="white")
    button.pack(pady=10)

    root.mainloop()

def on_button_click():
    print("Button clicked")

# Function to animate GIF at correct speed
def update_gif():
    global gif_frame_index
    gif_frame_index = (gif_frame_index + 1) % gif_image.n_frames  # Loop through frames
    gif_image.seek(gif_frame_index)  # Go to next frame
    gif_photo = ImageTk.PhotoImage(gif_image)  # Convert to Tkinter-compatible image
    gif_label.config(image=gif_photo)
    gif_label.image = gif_photo  # Prevent garbage collection

    # Get GIF frame delay (convert from milliseconds to an integer)
    delay = gif_image.info.get("duration", 100)  # Default to 100ms if missing
    splash.after(delay, update_gif)  # Adjust frame update timing

# Create splash screen
def create_splash_screen():
    global splash, gif_label, gif_image, gif_frame_index
    
    splash = tk.Tk()
    splash.title("Splash Screen")
    splash.geometry("1200x800")
    splash.configure(bg="white")

    # Load GIF
    gif_path = "plane_loading.gif"  # Change to your actual GIF file
    gif_image = Image.open(gif_path)
    gif_frame_index = 0

    # Label to display GIF
    gif_label = tk.Label(splash, bg="white")
    gif_label.pack(expand=True)

    # Start GIF animation
    update_gif()

    # Close splash screen after 5 seconds and open main window
    splash.after(5000, open_main_window)
    return splash

# Run splash screen
if __name__ == "__main__":
    splash = create_splash_screen()
    splash.mainloop()
