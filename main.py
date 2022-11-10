import uvicorn as uvicorn
from socketio import AsyncServer, ASGIApp

from recognition import detect

io = AsyncServer(async_mode="asgi",
                 cors_allowed_origins="*",
                 max_http_buffer_size=0xFFFFFFFF,
                 ping_timeout=300)
app = ASGIApp(io)


@io.event
def connect(sid, env, auth):
    print(f"[CONNECTED] Id: {sid}")


@io.event
def disconnect(sid):
    print(f"[DISCONNECTED] Id: {sid}")


@io.event
async def message(sid, msg):
    print(msg)


@io.on("*")
async def catch_all(event, sid, data):
    print(f"[{event}] {sid}: {data}")


@io.event
async def frame(sid, img):
    prediction = detect(img)
    await io.emit("frame", str(prediction))



