class LogLevel:
    """ Log level: info and error are always printed while debug
    is only for printing detailed messages. """
    INFO, DEBUG = range(2)


class Log:
    """ Class for logging information """

    f = None
    log_level = LogLevel.INFO
    disable = None

    @staticmethod
    def set_log_level(log_level):
        Log.log_level = log_level

    @staticmethod
    def open(file_name):
        Log.f = open(file_name, "w")

    @staticmethod
    def close():
        if Log.f is not None:
            Log.f.close()

    @staticmethod
    def debug(msg):
        if Log.f is not None and Log.log_level == LogLevel.DEBUG:
            Log.f.write(str(msg) + "\n")

    @staticmethod
    def info(msg):
        if Log.f is not None:
            Log.f.write(str(msg) + "\n")

    @staticmethod
    def error(msg):
        if Log.f is not None:
            Log.f.write(str(msg) + "\n")

    @staticmethod
    def flush():
        if Log.f is not None:
            Log.f.flush()
