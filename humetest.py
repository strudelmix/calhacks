import asyncio
import requests
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig
import os


from hume import HumeBatchClient
from hume.models.config import FaceConfig

base_dir = os.path.dirname(os.path.abspath(__file__))
test_img = os.path.join(base_dir, 'test.png')

client = HumeBatchClient("hyKUPUAgDDzg4lBNoHVbRpU4U6HtgTvtoX75cDZJrHkhplcE")
config = FaceConfig()
job = client.submit_job([test_img], [config])

print(job)
print("Running...")

url = "https://api.hume.ai/v0/batch/jobs"

payload = "{\"models\":{\"face\":{\"fps_pred\":3,\"prob_threshold\":0.99,\"identify_faces\":false,\"min_face_size\":60,\"save_faces\":false},\"prosody\":{\"granularity\":\"utterance\",\"window\":{\"length\":4,\"step\":1}},\"language\":{\"granularity\":\"word\"}},\"transcription\":{\"language\":null,\"identify_speakers\":false,\"confidence_threshold\":0.5},\"notify\":false}"
headers = {
    "accept": "application/json; charset=utf-8",
    "content-type": "application/json; charset=utf-8",
    "X-Hume-Api-Key": "hyKUPUAgDDzg4lBNoHVbRpU4U6HtgTvtoX75cDZJrHkhplcE"
}

response = requests.post(url, data=payload, headers=headers)

print(response.text)

details = job.await_complete()
# emotions = details["face"]["predictions"][0]["emotions"][0]["name"]
# print(emotions)
print(job.get_predictions())







import requests

url = "https://api.hume.ai/v0/batch/jobs"

payload = "{\"models\":{\"face\":{\"fps_pred\":3,\"prob_threshold\":0.99,\"identify_faces\":false,\"min_face_size\":60,\"save_faces\":false},\"prosody\":{\"granularity\":\"utterance\",\"window\":{\"length\":4,\"step\":1}},\"language\":{\"granularity\":\"word\"}},\"transcription\":{\"language\":null,\"identify_speakers\":false,\"confidence_threshold\":0.5},\"notify\":false}"
headers = {
    "accept": "application/json; charset=utf-8",
    "content-type": "application/json; charset=utf-8",
    "X-Hume-Api-Key": "hyKUPUAgDDzg4lBNoHVbRpU4U6HtgTvtoX75cDZJrHkhplcE"
}

response = requests.post(url, data=payload, headers=headers)

print(response.text)