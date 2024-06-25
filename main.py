from segmentation import Pool

from cv2 import waitKey
from datetime import date

if __name__ == "__main__":
    DATE = date.today().strftime("%d-%m-%Y")

    VIDEO_LINK_1 = 'dataset/output1.avi'
    VIDEO_LINK_2 = 'dataset/output2.avi'

    # rtsp://admin:1111@10.0.0.251/live/main 


    pools: list[Pool] = [
        Pool(src_link=VIDEO_LINK_1, pool_number=1, date=DATE),
        Pool(src_link=VIDEO_LINK_2, pool_number=2, date=DATE),
    ]

    while True:
        try:
            for pool in pools:
                pool.segment().show()
                if waitKey(10) == ord('q'):
                    break
        except Exception as err:
            print(err)