import socket
import io
import struct
import json

import torch
from PIL import Image
import numpy as np

from model import DirectMHPInfer
from utils.general import scale_coords


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DirectMHPInfer(weights="weights/agora_m_best.pt", device=device)

config = {
    "process_id": "directmhp",
    "server_address": "/tmp/gesurease.sock",
    "img_size": 640,
    "stride": model.model.stride.max().item(),
    "prediction": ["x1", "y1", "x2", "y2", "conf", "class", "pitch", "yaw", "roll"],
}


def pred(img):
    img = np.array(Image.open(io.BytesIO(img)).convert(mode="RGB"))

    img, old_shape = model.preprocess(img, config["img_size"], config["stride"])

    img = torch.from_numpy(img).to(device=device)
    img = img / 255.0

    img = img[None]

    start = time.time()
    out = model(img)[0]
    end = (time.time() - start) * 1000

    print(f"\t\tinference: {end:.1f} ms")

    out[:, :4] = scale_coords(img.shape[2:], out[:, :4].clone().detach(), old_shape[:2])

    out = [t.cpu().detach().numpy().tolist() for t in out]
    out = [dict(zip(config["prediction"], pred)) for pred in out]

    return json.dumps({"prediction": out})


def run():
    data_len_bytes = sock.recv(4)
    if len(data_len_bytes) == 0:
        print("Connection closed, exiting...")
        exit(1)

    data_len = struct.unpack("!I", data_len_bytes)[0]

    start = time.time()
    img = sock.recv(data_len)
    while len(img) < data_len:
        img += sock.recv(data_len - len(img))

    # print(img)

    start2 = time.time()
    preds = pred(img)
    end2 = (time.time() - start2) * 1000
    sock.sendall(struct.pack("!I", len(preds)))
    sock.sendall(preds.encode())
    end = (time.time() - start) * 1000
    print(f"\tipc time: {end - end2:.1f} ms")
    print(f"duration: {end:.1f} ms")


if __name__ == "__main__":
    import time

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(config["server_address"])
    sock.setblocking(True)

    # Send the process identifier to the Rust server
    sock.sendall(config["process_id"].encode())

    while True:
        run()
