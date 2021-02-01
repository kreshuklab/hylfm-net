class DuplicateLogFilter:
    def __init__(self):
        self.msgs = set()

    def __call__(self, record) -> int:
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return int(rv)
