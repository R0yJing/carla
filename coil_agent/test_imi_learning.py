from threading import Thread

b = False
def task():
    global b
    while not b:
        print("alive")
    while b:
        print("dead")
def func():
    global b
    b = True
t = Thread(target=task, daemon=True)
t.start()
import time

time.sleep(3)
func()
time.sleep(3)