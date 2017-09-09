import socket
import numpy as np


class ReliableConnect:

    DELIMITER = "<EOF>"
    BUFFER_SIZE = 1024
    ip_address = None
    port = None
    socket = None

    def __init__(self, ip_address, port, image_dim):
        self.ip_address = ip_address
        self.port = port
        self.socket = None
        self.row = image_dim
        self.col = image_dim
        self.channel = 4
        self.id = 0
        self.total_bytes = self.row * self.col * self.channel * 4

    def connect(self):
        # create an INET, STREAMing socket
        self.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)

        self.socket.connect((self.ip_address, self.port))

    def send_message(self, message):
        if self.socket is None:
            raise Exception("Socket is not initialized. Please connect before use")

        self.socket.send(message + ReliableConnect.DELIMITER)

    def close(self):
        self.socket.close()
        self.socket = None

    def receive_message(self):
        data = self.socket.recv(ReliableConnect.BUFFER_SIZE)
        return data

    def receive_image(self):
        """ Receives image over socket of size (ROW, COL, CHANNEL) """

        toread = self.total_bytes
        buf = bytearray(toread)
        view = memoryview(buf)
        
        self.id += 1

        while toread:
            nbytes = self.socket.recv_into(view, toread)
            view = view[nbytes:]  # slicing views is cheap
            toread -= nbytes
        img = np.frombuffer(buf, dtype='f4').reshape((self.row, self.col, self.channel))

        # Remove alpha channel
        img = img[:, :, :3]

        return img

    def send_and_receive_message(self, message):
        self.send_and_receive_message(message)
        return self.receive_message()
