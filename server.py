import tkinter as tk
from PIL import Image, ImageTk
import asyncio
import websockets
import base64
from io import BytesIO


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Камера")
        self.root.geometry("500x600")

        self.label = tk.Label(root, text="Ожидание изображения...")
        self.label.pack()

        self.canvas = tk.Canvas(root, width=400, height=400, bg="gray")
        self.canvas.pack()

        self.button = tk.Button(root, text="Запуск сервера", command=self.start_server)
        self.button.pack()

    def start_server(self):
        asyncio.run(self.run_server())

    async def receive_image(self, websocket, path):
        async for message in websocket:
            image_data = base64.b64decode(message)
            image = Image.open(BytesIO(image_data))
            image = image.resize((400, 400))
            self.display_image(image)

    def display_image(self, image):
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(200, 200, image=self.photo)

    async def run_server(self):
        async with websockets.serve(self.receive_image, "0.0.0.0", 8765):
            print("Сервер запущен на порту 8765")
            await asyncio.Future()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
