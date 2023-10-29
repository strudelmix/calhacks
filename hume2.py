import asyncio
import os
import hume
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig


async def main3():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_img = os.path.join(base_dir, 'test.png')
    client = HumeStreamClient("hyKUPUAgDDzg4lBNoHVbRpU4U6HtgTvtoX75cDZJrHkhplcE")
    configs = [FaceConfig(identify_faces=False)]
    async with client.connect(configs) as socket:
        socket: StreamSocket
        result = await socket.send_file(test_img)
        emotions = result["face"]["predictions"][0]["emotions"][0]["name"]
        # prob = result["face"]["predictions"][0]["emotions"]
        print(emotions)
        return emotions


asyncio.run(main3())
