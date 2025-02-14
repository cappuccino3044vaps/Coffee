#!/usr/bin/env python3.5
# -*- coding:utf-8 -*-

import serial
import time
import threading
import sys
from gpiozero import Button, DigitalOutputDevice

TRUE         =  1
FALSE        =  0

# Basic response message definition
ACK_SUCCESS           = 0x00
ACK_FAIL              = 0x01
ACK_FULL              = 0x04
ACK_NO_USER           = 0x05
ACK_TIMEOUT           = 0x08
ACK_GO_OUT            = 0x0F     # The center of the fingerprint is out of alignment with sensor

# User information definition
ACK_ALL_USER          = 0x00
ACK_GUEST_USER        = 0x01
ACK_NORMAL_USER       = 0x02
ACK_MASTER_USER       = 0x03

USER_MAX_CNT          = 1000        # Maximum fingerprint number

# Command definition
CMD_HEAD              = 0xF5
CMD_TAIL              = 0xF5
CMD_ADD_1             = 0x01
CMD_ADD_2             = 0x02
CMD_ADD_3             = 0x03
CMD_MATCH             = 0x0C
CMD_DEL               = 0x04
CMD_DEL_ALL           = 0x05
CMD_USER_CNT          = 0x09
CMD_COM_LEV           = 0x28
CMD_LP_MODE           = 0x2C
CMD_TIMEOUT           = 0x2E

CMD_FINGER_DETECTED   = 0x14

# ピン定義 (BCM番号)
Finger_WAKE_Pin   = 23  # 入力: 指紋検出用
Finger_RST_Pin    = 24  # 出力: リセット／スリープ制御用

# gpiozero によるピン初期化
finger_wake = Button(Finger_WAKE_Pin)
finger_rst  = DigitalOutputDevice(Finger_RST_Pin, initial_value=True)  # 初期状態: HIGH

g_rx_buf            = []
PC_Command_RxBuf    = []
Finger_SleepFlag    = 0

ser = serial.Serial("/dev/ttyS0", 19200)

#***************************************************************************
# @brief    send a command, and wait for the response of module
#***************************************************************************
def TxAndRxCmd(command_buf, rx_bytes_need, timeout):
    global g_rx_buf
    CheckSum = 0
    tx_buf = []
    tx = ""
    
    tx_buf.append(CMD_HEAD)
    for byte in command_buf:
        tx_buf.append(byte)
        CheckSum ^= byte
        
    tx_buf.append(CheckSum)
    tx_buf.append(CMD_TAIL)
    
    for i in tx_buf:
        tx += chr(i)
        
    ser.flushInput()
    ser.write(tx)
    
    g_rx_buf = []
    time_before = time.time()
    time_after = time.time()
    while time_after - time_before < timeout and len(g_rx_buf) < rx_bytes_need:  # Waiting for response
        bytes_can_recv = ser.inWaiting()
        if bytes_can_recv != 0:
            g_rx_buf += ser.read(bytes_can_recv)
        time_after = time.time()

    for i in range(len(g_rx_buf)):
        g_rx_buf[i] = ord(g_rx_buf[i])

    if len(g_rx_buf) != rx_bytes_need:
        return ACK_TIMEOUT
    if g_rx_buf[0] != CMD_HEAD:
        return ACK_FAIL
    if g_rx_buf[rx_bytes_need - 1] != CMD_TAIL:
        return ACK_FAIL
    if g_rx_buf[1] != tx_buf[1]:
        return ACK_FAIL

    CheckSum = 0
    for index, byte in enumerate(g_rx_buf):
        if index == 0:
            continue
        if index == 6:
            if CheckSum != byte:
                return ACK_FAIL
        CheckSum ^= byte
    return ACK_SUCCESS

#***************************************************************************
# @brief    Get Compare Level
#***************************************************************************    
def GetCompareLevel():
    global g_rx_buf
    command_buf = [CMD_COM_LEV, 0, 0, 1, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief    Set Compare Level, the default value is 5, 
#           can be set to 0-9, the bigger, the stricter
#***************************************************************************
def SetCompareLevel(level):
    global g_rx_buf
    command_buf = [CMD_COM_LEV, 0, level, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief    Query the number of existing fingerprints
#***************************************************************************
def GetUserCount():
    global g_rx_buf
    command_buf = [CMD_USER_CNT, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief    Get the time that fingerprint collection wait timeout
#***************************************************************************        
def GetTimeOut():
    global g_rx_buf
    command_buf = [CMD_TIMEOUT, 0, 0, 1, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF

#***************************************************************************
# @brief    Register fingerprint
#***************************************************************************
def AddUser():
    global g_rx_buf
    r = GetUserCount()
    if r >= USER_MAX_CNT:
        return ACK_FULL
        
    command_buf = [CMD_ADD_1, 0, r+1, 3, 0]
    print(g_rx_buf)
    r = TxAndRxCmd(command_buf, 8, 6)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        command_buf[0] = CMD_ADD_3
        r = TxAndRxCmd(command_buf, 8, 6)
        print(g_rx_buf)
        if r == ACK_TIMEOUT:
            return ACK_TIMEOUT
        if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
            return ACK_SUCCESS
        else:
            return ACK_FAIL
    else:
        return ACK_FAIL

#***************************************************************************
# @brief    Clear fingerprints
#***************************************************************************
def ClearAllUser():
    global g_rx_buf
    command_buf = [CMD_DEL_ALL, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return ACK_SUCCESS
    else:
        return ACK_FAIL

#***************************************************************************
# @brief    Check if user ID is between 1 and 3
#***************************************************************************
def IsMasterUser(user_id):
    if user_id == 1 or user_id == 2 or user_id == 3:
        return TRUE
    else:
        return FALSE

#***************************************************************************
# @brief    Fingerprint matching
#***************************************************************************
def VerifyUser():
    global g_rx_buf
    command_buf = [CMD_MATCH, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and IsMasterUser(g_rx_buf[4]) == TRUE:
        return ACK_SUCCESS
    elif g_rx_buf[4] == ACK_NO_USER:
        return ACK_NO_USER
    elif g_rx_buf[4] == ACK_TIMEOUT:
        return ACK_TIMEOUT
    else:
        return ACK_GO_OUT

#***************************************************************************
# @brief    Analysis the command from PC terminal
#***************************************************************************
def Analysis_PC_Command(command):
    global Finger_SleepFlag
    
    if command == "CMD1" and Finger_SleepFlag != 1:
        print("Number of fingerprints already available:  %d" % GetUserCount())
    elif command == "CMD2" and Finger_SleepFlag != 1:
        print("Add fingerprint  (Put your finger on sensor until successfully/failed information returned)")
        r = AddUser()
        if r == ACK_SUCCESS:
            print("Fingerprint added successfully !")
        elif r == ACK_FAIL:
            print("Failed: Please try to place the center of the fingerprint flat to sensor, or this fingerprint already exists !")
        elif r == ACK_FULL:
            print("Failed: The fingerprint library is full !")
    elif command == "CMD3" and Finger_SleepFlag != 1:
        print("Waiting Finger......Please try to place the center of the fingerprint flat to sensor !")
        r = VerifyUser()
        if r == ACK_SUCCESS:
            print("Matching successful !")
        elif r == ACK_NO_USER:
            print("Failed: This fingerprint was not found in the library !")
        elif r == ACK_TIMEOUT:
            print("Failed: Time out !")
        elif r == ACK_GO_OUT:
            print("Failed: Please try to place the center of the fingerprint flat to sensor !")
    elif command == "CMD4" and Finger_SleepFlag != 1:
        ClearAllUser()
        print("All fingerprints have been cleared !")
    elif command == "CMD5" and Finger_SleepFlag != 1:
        # スリープモードへ：RSTを LOW にしてモジュールを停止
        finger_rst.off()
        Finger_SleepFlag = 1
        print("Module has entered sleep mode: you can use the finger Automatic wake-up function, in this mode, only CMD6 is valid, send CMD6 to pull up the RST pin of module, so that the module exits sleep !")
    elif command == "CMD6":
        Finger_SleepFlag = 0
        finger_rst.on()
        print("The module is awake. All commands are valid !")
    else:
        print("commands are invalid !")

#***************************************************************************
# @brief   If you enter the sleep mode, then open the Automatic wake-up function of the finger,
#          begin to check if the finger is pressed, and then start the module and match
#***************************************************************************
def Auto_Verify_Finger():
    while True:
        if Finger_SleepFlag == 1:
            if finger_wake.is_active:  # 指が検出された場合
                time.sleep(0.01)
                if finger_wake.is_active:
                    # RSTピンを HIGH にしてモジュール起動
                    finger_rst.on()
                    time.sleep(0.25)  # モジュール起動待ち
                    print("Waiting Finger......Please try to place the center of the fingerprint flat to sensor !")
        time.sleep(0.2)

def main():
    # モジュールの初期化: 一度リセット（RSTをLOW→HIGH）
    finger_rst.off()
    time.sleep(0.25)
    finger_rst.on()
    time.sleep(0.25)  # モジュール起動待ち
    
    while SetCompareLevel(5) != 5:
        print("***ERROR***: Please ensure that the module power supply is 3.3V or 5V, the serial line connection is correct.")
        time.sleep(1)
    
    print("***************************** WaveShare Capacitive Fingerprint Reader Test *****************************")
    print("Compare Level:  5    (can be set to 0-9, the bigger, the stricter)")
    print("Number of fingerprints already available:  %d " % GetUserCount())
    print(" send commands to operate the module: ")
    print("  CMD1 : Query the number of existing fingerprints")
    print("  CMD2 : Registered fingerprint  (Put your finger on the sensor until successfully/failed information returned)")
    print("  CMD3 : Fingerprint matching  (Send the command, put your finger on sensor)")
    print("  CMD4 : Clear fingerprints")
    print("  CMD5 : Switch to sleep mode, you can use the finger Automatic wake-up function (In this state, only CMD6 is valid. When a finger is placed on the sensor, the module is awakened and the finger is matched, without sending commands to match each time. The CMD6 can be used to wake up)")
    print("  CMD6 : Wake up and make all commands valid")
    print("***************************** WaveShare Capacitive Fingerprint Reader Test *****************************")

    t = threading.Thread(target=Auto_Verify_Finger)
    t.setDaemon(True)
    t.start()
    
    while True:
        # Python 3 では input() を使用
        command = input("Please input command (CMD1-CMD6):")
        Analysis_PC_Command(command)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        if ser is not None:
            ser.close()
        # gpiozero は自動でクリーンアップされますが、必要に応じて close() を呼び出すことも可能です
        finger_rst.close()
        finger_wake.close()
        print("\n\n Test finished ! \n")
        sys.exit()