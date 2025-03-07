from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
import asyncio
import websockets
import base64
from threading import Thread

#123123
class CameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        self.image = Image()
        self.layout.add_widget(self.image)

        self.button = Button(text='Подключиться', size_hint=(1, 0.2))
        self.button.bind(on_press=self.start_stream)
        self.layout.add_widget(self.button)

        return self.layout

    def start_stream(self, instance):
        self.capture = cv2.VideoCapture(0)
        Thread(target=self.send_frames, daemon=True).start()

    async def send_image(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = base64.b64encode(buffer).decode('utf-8')
        async with websockets.connect('ws://176.59.46.104:8765') as websocket:
            await websocket.send(image_data)

    def send_frames(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame, 0)
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
                self.image.texture = texture
                asyncio.run(self.send_image(frame))


if __name__ == '__main__':
    CameraApp().run()
