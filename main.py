from datetime import date

from cv2 import waitKey

from segmentation import Pool

if __name__ == "__main__":
    DATE = date.today().strftime("%d-%m-%Y")

    # VIDEO_LINK_1 = "rtsp://admin:1111@10.0.0.251/live/main"
    VIDEO_LINK_1 = "rtsp://pool251:251_pool@45.152.168.61:52094"
    pools: list[Pool] = [
        Pool(src_link=VIDEO_LINK_1, pool_number=1, date=DATE),
    ]

    while True:
        try:
            for pool in pools:
                pool.segment().show()
                if waitKey(10) == ord("q"):
                    break
        except Exception as err:
            print(err)
