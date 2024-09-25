from datetime import date
from threading import Thread

import cv2 as cv
import numpy as np

from vanischeCV import *

# TO-DO: fill out large contours

MIN_AREA = 4500
MAX_AREA = 9000


class Pool:
    H, W = 740, 740

    POOL_ROI = ROI(
        76,
        1036,
        431,
        1391,
    )

    INFO_TABLE_MASK_ROI = ROI(
        y1=H // 4 * 3,
        y2=H,
        x1=0,
        x2=H // 4,
    )
    INFO_TABLE_GRAY_ROI = ROI(
        y1=H // 2,
        y2=H // 4 * 3,
        x1=0,
        x2=H // 4,
    )
    INFO_TABLE_HSV_ROI = ROI(
        y1=H // 4,
        y2=H // 2,
        x1=0,
        x2=H // 4,
    )

    CROPPING_MASK = Frame(cv.imread("dataset/cropping_mask.jpg"), "gray")
    INFO_TABLE = Frame(cv.imread("dataset/info_table.jpg"), "gray")

    def __init__(self, src_link: str | int, pool_number: int, date: str) -> None:
        self.src_link = src_link
        self.pool_number = pool_number
        self.date = date.replace("/", ".")

        self.cap = cv.VideoCapture(self.src_link)
        self.winname = f"Pool {self.pool_number}"

        self.sturgeons_areas_array = np.array([])
        self.sturgeons_average_area: int = 0

        Frame.create_thresh_tb_window(winname=self.winname, default_value=65)

    def get_frame(self) -> Frame:
        ret, frame = self.cap.read()
        assert ret, f"Couldnt get frame from {self.src_link}"

        self.raw = (
            Frame(frame, "bgr")
            .roi(self.POOL_ROI)
            .roi(ROI(110, 850, 110, 850))
            .bitwise(self.CROPPING_MASK)
        )
        self.hsv = self.raw.cvt2hsv()
        self.gray = self.raw.cvt2gray()

        return self.raw

    def get_sturgeons_mask(self) -> Frame:
        self.bgr_mask, self.mask = self.raw.thresh_tb_mask(
            self.raw.cvt2gray(),
            self.winname,
            invert=True,
            mask_color=Colors.GREEN,
        )

        return self.bgr_mask, self.mask

    def get_sturgeons_conts(self, mask: Frame) -> list[Contour]:
        self.sturgeons: list[Contour] = []

        for c in mask.get_conts().conts:
            cont = Contour(c).approx(value=0.04)
            if MIN_AREA < cont.get_area() < MAX_AREA:
                cont.get_moments()
                cont.get_m_center()
                cont.get_bounding_rect()

                if not (
                    abs(self.mask.w // 2 - (cont.x + cont.rect.w // 2)) < 10
                    and abs(self.mask.h // 2 - (cont.y + cont.rect.h // 2)) < 10
                ):
                    self.sturgeons.append(cont)
        return self.sturgeons

    def fill_info_table(self) -> Frame:
        self.info_table = Frame(cv.imread("dataset/info_table.jpg"), "bgr")

        self.info_table.put_roi(
            self.mask.resize(self.H // 4, self.W // 4).invert(),
            self.INFO_TABLE_MASK_ROI,
        )
        self.info_table.put_roi(
            self.gray.resize(self.H // 4, self.W // 4), self.INFO_TABLE_GRAY_ROI
        )
        self.info_table.put_roi(
            self.hsv.resize(self.H // 4, self.W // 4), self.INFO_TABLE_HSV_ROI
        )

        self.info_table.print(
            self.sturgeons_average_area,
            Point(10, 100),
            scale=1.2,
            thickness=2,
        )
        for i in range(len(self.date)):
            if i == "-":
                self.date[i] = "."
        self.info_table.print(
            self.date,
            Point(10, 40),
            scale=0.9,
            thickness=2,
        )

        return self.info_table

    def parse_sturgeons(self) -> Frame:
        self.frame = self.raw.bitwise(self.CROPPING_MASK).bitwise(self.bgr_mask)

        if self.sturgeons:
            for sturgeon in self.sturgeons:
                self.frame.draw_rect(sturgeon.rect, thickness=1, color=Colors.BLACK)
                self.frame.draw_conts(sturgeon, thickness=4, color=Colors.GREEN)
                self.frame.draw_point(sturgeon.m_center)

                self.sturgeons_areas_array = np.append(
                    self.sturgeons_areas_array, sturgeon.area
                )

        self.sturgeons_average_area = int(np.mean(self.sturgeons_areas_array))

    def segment(self) -> object:
        self.get_frame()

        self.get_sturgeons_mask()
        self.get_sturgeons_conts(self.mask)
        self.parse_sturgeons()

        self.fill_info_table()

        self.output_frame = Frame(
            cv.hconcat([self.info_table.src, self.frame.src]), "bgr"
        ).resize(625, 500)

        return self

    def show(self) -> None:
        cv.imshow(self.winname, self.output_frame.src)


if __name__ == "__main__":
    DATE = date.today().strftime("%d/%m/%Y").replace("-", ".")

    SRC_LINK_1 = "dataset/output1.avi"
    SRC_LINK_2 = "dataset/output2.avi"

    pools: list[Pool] = [
        Pool(src_link=SRC_LINK_1, pool_number=1, date=DATE),
        Pool(src_link=SRC_LINK_2, pool_number=2, date=DATE),
    ]

    while True:
        try:
            for pool in pools:
                pool.segment().show()

                cv.waitKey(10)
        except Exception as err:
            print(err)
