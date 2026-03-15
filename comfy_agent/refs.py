class DataRef:
    """Reference to a node output"""

    def __init__(self, node_id, output_index):
        self.node_id = str(node_id)
        self.output_index = int(output_index)

    def as_tuple(self):
        return (self.node_id, self.output_index)

    def __repr__(self):
        return f"DataRef(node_id={self.node_id!r}, output_index={self.output_index})"