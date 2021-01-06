import sys, select, termios, tty
import time

moveBindings = {
    'a': 0,
    'd': 1,
    's': 2,
    'w': 3,
    'x': 4,
    'q': 5,
    'e': 6
}


class KEYBOARD:
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)

    def get_key(self, total_times=0.01):
        start = time.time()
        key_final = None
        while time.time() - start < total_times:
            a = self.get_single_key(None)
            if a is not None:
                key_final = a
        return key_final

    def get_single_key(self, key_timeout):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        if key in moveBindings.keys():
            return moveBindings[key]
        else:
            return None


if __name__ == "__main__":
    keyboard = KEYBOARD()
    while True:
        k = keyboard.get_key()
        print(type(k))
        if k is None:
            print("None")
        else:
            print(k)
        # print(key.getKey(None))

