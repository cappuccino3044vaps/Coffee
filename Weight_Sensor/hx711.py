from gpiozero import DigitalInputDevice, DigitalOutputDevice
import time

# GPIOピンの設定
HX711_DAT_PIN = 5  # GPIO5 (Pin29)
HX711_CLK_PIN = 6  # GPIO6 (Pin31)

data_pin = DigitalInputDevice(HX711_DAT_PIN, pull_up=True)
clock_pin = DigitalOutputDevice(HX711_CLK_PIN)

def read_hx711():
    # センサが準備できるまで待機（data_pinが LOW になるまで）
    while data_pin.value:
        time.sleep(0.001)
    reading = 0
    # 24ビット分、センサからデータを読み込む
    for _ in range(24):
        clock_pin.on()
        time.sleep(0.000001)
        reading = reading << 1
        clock_pin.off()
        time.sleep(0.000001)
        if data_pin.value:
            reading += 1
    # ゲイン設定用の1パルス（必要に応じて調整）
    clock_pin.on()
    time.sleep(0.000001)
    clock_pin.off()
    # 24ビットの符号付き値へ変換
    if reading & 0x800000:
        reading = -((~reading & 0xffffff) + 1)
    return reading

def main():
    while True:
        value = read_hx711()
        print("HX711 reading:", value)
        time.sleep(1)

if __name__ == '__main__':
    main()
