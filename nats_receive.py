import cv2
import numpy as np
import asyncio
from nats.aio.client import Client as NATS

# NATS 서버 주소 설정 (NATS 서버를 배포한 호스트 머신의 공인 IP로 수정)
nats_server = "nats://(IP Address):4222"

async def receive_message():
    # 새 이벤트 루프 생성 및 설정
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    nc = NATS()
    await nc.connect(nats_server)

    async def video_handler(msg):
        # print(f"Received a message: {msg.data.decode()}")
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(480, 640, 3)

        # 실시간 영상 출력
        cv2.imshow('HPC Server Video Stream', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            await nc.close()
            cv2.destroyAllWindows()
            return

    await nc.subscribe("video.stream", cb=video_handler)
    
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Subscription cancelled.")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(receive_message())
    finally:
        loop.close()
