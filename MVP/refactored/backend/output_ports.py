class Output:
    def __init__(self, id, x=0, y=0, movable=False):
        self.id = id
        self.generator = None
        self.wire = None

        self.x = x
        self.y = y
        self.movable = movable

    def set_position(self, x, y):
        self.x = x
        self.y = y