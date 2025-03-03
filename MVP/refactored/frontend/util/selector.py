from MVP.refactored.frontend.canvas_objects.box import Box
from MVP.refactored.frontend.canvas_objects.spider import Spider
import copy


class Selector:
    def __init__(self, main_diagram, canvas=None):
        if canvas is None:
            self.canvas = main_diagram.custom_canvas
        else:
            self.canvas = canvas
        self.selecting = False
        self.selected_items = []
        self.selected_boxes = []
        self.selected_spiders = []
        self.selected_wires = []
        self.origin_x = None
        self.origin_y = None
        self.copied_items = []
        self.copied_wire_list = []
        self.copied_left_wires = []
        self.copied_right_wires = []

    def start_selection(self, event):
        for item in self.selected_items:
            if item not in self.canvas.search_result_highlights:
                item.deselect()
            else:
                item.search_highlight()
        self.selecting = True
        self.origin_x = event.x
        self.origin_y = event.y
        self.canvas.selectBox = self.canvas.create_rectangle(self.origin_x, self.origin_y, self.origin_x + 1,
                                                             self.origin_y + 1)
        self.selected_items.clear()
        self.selected_boxes.clear()
        self.selected_spiders.clear()
        self.selected_wires.clear()

    def update_selection(self, event):
        if self.selecting:
            x_new = event.x
            y_new = event.y
            self.canvas.coords(self.canvas.selectBox, self.origin_x, self.origin_y, x_new, y_new)

    def finalize_selection(self, boxes, spiders, wires):
        if self.selecting:
            selected_coordinates = self.canvas.coords(self.canvas.selectBox)

            self.selected_boxes = [box for box in boxes if self.is_within_selection(box.rect, selected_coordinates)]

            self.selected_spiders = [spider for spider in spiders if
                                     self.is_within_selection_point(spider.location, selected_coordinates)]

            self.selected_wires = [wire for wire in wires if
                                   self.is_within_selection_point(wire.start_connection.location, selected_coordinates)
                                   or self.is_within_selection_point(wire.end_connection.location,
                                                                     selected_coordinates)]
            self.selected_items = self.selected_boxes + self.selected_spiders + self.selected_wires
            for item in self.selected_items:
                item.select()

    def select_action(self):
        if self.selecting:
            self.canvas.delete(self.canvas.selectBox)
            self.selecting = False

    def finish_selection(self):
        for item in self.selected_items:
            if item not in self.canvas.search_result_highlights:
                item.deselect()
            else:
                main_diagram = self.canvas.main_diagram
                active = main_diagram.search_results[main_diagram.active_search_index]
                active_items = [main_diagram.search_objects[index] for index in active]
                if item in active_items:
                    item.search_highlight_primary()
                else:
                    item.search_highlight_secondary()
        self.selected_items.clear()

        self.canvas.delete(self.canvas.selectBox)
        self.selecting = False

    def create_sub_diagram(self):
        coordinates = self.find_corners_selected_items()
        if len(self.selected_boxes) == 0 and len(self.selected_spiders) == 0:
            return
        x = (coordinates[0] + coordinates[2]) / 2
        y = (coordinates[1] + coordinates[3]) / 2
        box = self.canvas.add_box((x, y), shape="rectangle")
        for wire in filter(lambda w: w in self.canvas.wires, self.selected_wires):
            wire.delete_self("sub_diagram")
        for box_ in filter(lambda b: b in self.canvas.boxes, self.selected_boxes):
            box_.delete_box(keep_sub_diagram=True, action="sub_diagram")
        for spider in filter(lambda s: s in self.canvas.spiders, self.selected_spiders):
            spider.delete_spider("sub_diagram")
            if self.canvas.receiver.listener:
                self.canvas.receiver.receiver_callback(
                    'create_spider_parent', wire_id=spider.id, connection_id=spider.id, generator_id=box.id
                )
        sub_diagram = box.edit_sub_diagram(save_to_canvasses=False)
        prev_status = self.canvas.receiver.listener
        self.canvas.receiver.listener = False
        self.canvas.copier.copy_canvas_contents(
            sub_diagram, self.selected_wires, self.selected_boxes, self.selected_spiders, coordinates, box
        )
        box.lock_box()
        self.canvas.receiver.listener = prev_status

        sub_diagram.set_name(str(sub_diagram.id)[-6:])
        box.set_label(str(sub_diagram.id)[-6:])
        self.canvas.main_diagram.add_canvas(sub_diagram)

    def is_within_selection(self, rect, selection_coords):
        if len(self.canvas.coords(rect)) == 4:
            x1, y1, x2, y2 = self.canvas.coords(rect)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
        elif len(self.canvas.coords(rect)) == 6:
            x1, y1, x2, y2, x3, y3 = self.canvas.coords(rect)
            x = (x1 + x2 + x3) / 3
            y = (y1 + y2 + y3) / 3
        else:
            return False
        return selection_coords[0] <= x <= selection_coords[2] and selection_coords[1] <= y <= selection_coords[3]

    def delete_selected_items(self):
        for item in self.selected_items:
            if isinstance(item, Box):
                if item.sub_diagram:
                    action_param = "sub_diagram"
                else:
                    action_param = None
                item.delete_box(action=action_param)
            if isinstance(item, Spider):
                item.delete_spider()
        self.selected_items.clear()

    @staticmethod
    def is_within_selection_point(point, selection_coords):
        """Check if a point is within the selection area."""
        x, y = point
        return selection_coords[0] <= x <= selection_coords[2] and selection_coords[1] <= y <= selection_coords[3]

    def copy_selected_items(self, canvas=None):
        if len(self.selected_items) > 0:
            self.copied_items.clear()
            self.copied_wire_list.clear()
            self.copied_left_wires.clear()
            self.copied_right_wires.clear()
            connection_list = []

            for item in self.selected_items:
                if isinstance(item, Box):
                    connections_copy = []
                    for connection in item.connections:
                        connection_list.append(connection)
                        connections_copy.append({
                            'id': copy.deepcopy(connection.id),
                            'side': copy.deepcopy(connection.side),
                            'index': copy.deepcopy(connection.index)
                        })

                    self.copied_items.append({
                        'component': "Box",
                        'id': copy.deepcopy(item.id),
                        'label': copy.deepcopy(item.label_text),
                        'location': (item.x, item.y),
                        'size': copy.deepcopy(item.size),
                        'shape': copy.deepcopy(item.shape),
                        'connections': connections_copy,
                        'sub-diagram': copy.deepcopy(item.sub_diagram.id) if item.sub_diagram else None
                    })
                    if item.sub_diagram:
                        if item.label_text not in self.canvas.master.project_exporter.get_current_d():
                            self.canvas.master.save_box_to_diagram_menu(item)

                if isinstance(item, Spider):
                    for wire in item.wires:
                        if wire.start_connection == item:
                            connection_list.append(wire.start_connection)
                        else:
                            connection_list.append(wire.end_connection)
                    self.copied_items.append({
                        'component': "Spider",
                        'id': copy.deepcopy(item.id),
                        'location': (item.x, item.y),
                        'size': copy.deepcopy(item.r),
                    })
            self.copy_selected_wires(connection_list, canvas)

    def paste_copied_items(self, event_x=50, event_y=50, replace=False, multi=1):
        if len(self.copied_items) > 0:

            middle_point = self.find_middle_point(event_x, event_y)
            wires = self.copied_left_wires + self.copied_right_wires
            pasted_items = []

            for item in self.copied_items:
                if item['component'] == "Box":
                    loc = (event_x + (item['location'][0] - middle_point[0]) * multi,
                           event_y + (item['location'][1] - middle_point[1]) * multi)
                    if item["sub-diagram"]:
                        new_box = self.canvas.main_diagram.importer.add_box_from_menu(
                            self.canvas, item['label'], loc, True)
                        new_box.update_size(new_box.size[0] * multi, new_box.size[1] * multi)
                        new_box.move_label()
                    else:
                        new_box = self.canvas.add_box(loc, size=(item['size'][0] * multi, item['size'][1] * multi),
                                                      shape=item['shape'])
                        for c in item['connections']:
                            if c['side'] == "right":
                                new_box.add_right_connection()
                            if c['side'] == "left":
                                new_box.add_left_connection()
                        new_box.set_label(item['label'])

                    for wire in self.copied_wire_list:
                        for box_connection in item['connections']:
                            if wire['original_start_connection'] == box_connection['id']:
                                for connection in new_box.connections:
                                    if (connection.side == wire['original_start_side']
                                            and connection.index == wire['original_start_index']):
                                        wire['start_connection'] = connection
                            if wire['original_end_connection'] == box_connection['id']:
                                for connection in new_box.connections:
                                    if (connection.side == wire['original_end_side']
                                            and connection.index == wire['original_end_index']):
                                        wire['end_connection'] = connection
                    if replace:
                        for wire in wires:
                            for box_connection in item['connections']:
                                if wire['original_start_connection'] == box_connection['id']:
                                    for connection in new_box.connections:
                                        if (connection.side == wire['original_start_side']
                                                and connection.index == wire['original_start_index']):
                                            wire['start_connection'] = connection
                    pasted_items.append(new_box)
                if item['component'] == "Spider":
                    new_spider = self.canvas.add_spider((event_x + (item['location'][0] - middle_point[0]) * multi,
                                                         event_y + (item['location'][1] - middle_point[1]) * multi))
                    pasted_items.append(new_spider)
                    for wire in self.copied_wire_list:
                        if wire['original_start_connection'] == item['id']:
                            wire['start_connection'] = new_spider
                        if wire['original_end_connection'] == item['id']:
                            wire['end_connection'] = new_spider
                    if replace:
                        for wire in wires:
                            if wire['original_start_connection'] == item['id']:
                                wire['start_connection'] = new_spider

            for wire in self.copied_wire_list:
                self.canvas.start_wire_from_connection(wire['start_connection'])
                self.canvas.end_wire_to_connection(wire['end_connection'])
            if replace:
                self.add_edge_wires(pasted_items)

    def find_middle_point(self, event_x, event_y):
        most_left, most_right, most_up, most_down = self.find_corners_copied_items()

        middle_x = (most_left + most_right) / 2
        middle_y = (most_up + most_down) / 2

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if most_left - (middle_x - event_x) < self.canvas.canvasx(0):
            middle_x = event_x + most_left - self.canvas.canvasx(0)
        if most_right - (middle_x - event_x) > self.canvas.canvasx(canvas_width):
            middle_x = event_x - (self.canvas.canvasx(canvas_width) - most_right)

        if most_up - (middle_y - event_y) < self.canvas.canvasy(0):
            middle_y = event_y + most_up - self.canvas.canvasy(0)
        if most_down - (middle_y - event_y) > self.canvas.canvasy(canvas_height):
            middle_y = event_y - (self.canvas.canvasy(canvas_height) - most_down)

        return middle_x, middle_y

    def find_corners_copied_items(self):
        most_left = self.canvas.winfo_width()
        most_right = 0
        most_up = self.canvas.winfo_height()
        most_down = 0
        for item in self.copied_items:
            if item['component'] == "Box":
                if item['location'][0] < most_left:
                    most_left = item['location'][0]
                if item['location'][1] < most_up:
                    most_up = item['location'][1]
                if item['location'][0] + item['size'][0] > most_right:
                    most_right = item['location'][0] + item['size'][0]
                if item['location'][1] + item['size'][1] > most_down:
                    most_down = item['location'][1] + item['size'][1]
            if item['component'] == "Spider":
                if item['location'][0] - item['size'] < most_left:
                    most_left = item['location'][0] - item['size']
                if item['location'][1] - item['size'] < most_up:
                    most_up = item['location'][1] - item['size']
                if item['location'][0] + item['size'] > most_right:
                    most_right = item['location'][0] + item['size']
                if item['location'][1] + item['size'] > most_down:
                    most_down = item['location'][1] + item['size']
        return most_left, most_right, most_up, most_down,

    def find_corners_selected_items(self):
        most_left = self.canvas.winfo_width()
        most_right = 0
        most_up = self.canvas.winfo_height()
        most_down = 0
        for item in self.selected_items:
            if isinstance(item, Box):
                if item.x < most_left:
                    most_left = item.x
                if item.y < most_up:
                    most_up = item.y
                if item.x + item.size[0] > most_right:
                    most_right = item.x + item.size[0]
                if item.y + item.size[1] > most_down:
                    most_down = item.y + item.size[1]
            if isinstance(item, Spider):
                if item.x - item.r < most_left:
                    most_left = item.x - item.r
                if item.y - item.r < most_up:
                    most_up = item.y - item.r
                if item.x + item.r > most_right:
                    most_right = item.x + item.r
                if item.y + item.r > most_down:
                    most_down = item.y + item.r
        return [most_left, most_up, most_right, most_down]

    def select_wires_between_selected_items(self):
        self.selected_boxes.clear()
        self.selected_spiders.clear()
        self.selected_wires.clear()
        selected_connections = []
        for item in self.selected_items:
            if isinstance(item, Box):
                for connection in item.connections:
                    selected_connections.append(connection)
            if isinstance(item, Spider):
                selected_connections.append(item)
        for wire in self.canvas.wires:
            if wire.start_connection in selected_connections and wire.end_connection in selected_connections:
                if wire not in self.canvas.selector.selected_items:
                    wire.select()
                    self.canvas.selector.selected_items.append(wire)
            if wire.start_connection not in selected_connections or wire.end_connection not in selected_connections:
                if wire in self.canvas.selector.selected_items:
                    wire.deselect()
                    self.canvas.selector.selected_items.remove(wire)

    def find_side_connections(self):
        connection_list = []
        left_wires = []
        right_wires = []
        wires = []
        connections_to_connect_left = []
        connections_to_connect_right = []
        # Find all connections
        for item in self.selected_items:
            if isinstance(item, Box):
                for connection in item.connections:
                    connection_list.append(connection)
            if isinstance(item, Spider):
                for wire in item.wires:
                    if wire.start_connection == item:
                        connection_list.append(wire.start_connection)
                    else:
                        connection_list.append(wire.end_connection)
        # Find wires with only 1 connection in connection_list (not fully in copied/selected items)
        for wire in self.canvas.wires:
            if wire.start_connection in connection_list and wire.end_connection not in connection_list:
                wires.append(wire)
            if wire.start_connection not in connection_list and wire.end_connection in connection_list:
                wires.append(wire)
        # Find side where wire goes outside selected items
        for wire in wires:
            if wire.start_connection in connection_list:
                self.categorize_wire(wire, wire.start_connection, left_wires, right_wires)
            if wire.end_connection in connection_list:
                self.categorize_wire(wire, wire.end_connection, left_wires, right_wires)
        # Sort wires based on connection height
        right_wires.sort(
            key=lambda w:
            w.end_connection.location[1] if w.end_connection in connection_list else w.start_connection.location[1])
        left_wires.sort(
            key=lambda w:
            w.end_connection.location[1] if w.end_connection in connection_list else w.start_connection.location[1])
        # Add connection to connect to lists
        for wire in left_wires:
            if wire.start_connection in connection_list:
                connections_to_connect_left.append(wire.end_connection)
            else:
                connections_to_connect_left.append(wire.start_connection)
        for wire in right_wires:
            if wire.start_connection in connection_list:
                connections_to_connect_right.append(wire.end_connection)
            else:
                connections_to_connect_right.append(wire.start_connection)
        return connections_to_connect_left, connections_to_connect_right

    def add_copied_wire(self, connection, is_left):
        wire_data = {
            'wire': "Wire",
            'start_connection': None,
            'end_connection': "New",
            'original_start_connection': copy.deepcopy(connection.id),
            'original_start_index': copy.deepcopy(connection.index),
            'original_start_side': copy.deepcopy(connection.side)
        }
        if is_left:
            self.copied_left_wires.append(wire_data)
        else:
            self.copied_right_wires.append(wire_data)

    def reconnect_wires(self, copied_wires, connections):
        min_length = min(len(copied_wires), len(connections))
        for i in range(min_length):
            self.canvas.start_wire_from_connection(copied_wires[i]['start_connection'])
            self.canvas.end_wire_to_connection(connections[i])

    @staticmethod
    def categorize_wire(wire, connection, left_wires, right_wires):
        if connection.side == "left":
            left_wires.append(wire)
        elif connection.side == "right":
            right_wires.append(wire)
        elif connection.side == "spider":
            if connection == wire.start_connection:
                if connection.location[0] > wire.end_connection.location[0]:
                    left_wires.append(wire)
                else:
                    right_wires.append(wire)
            if connection == wire.end_connection:
                if connection.location[0] > wire.start_connection.location[0]:
                    left_wires.append(wire)
                else:
                    right_wires.append(wire)

    @staticmethod
    def find_edge_items(items):
        most_left = []
        most_left_distance = float('inf')
        most_right = []
        most_right_distance = 0
        for item in items:
            if isinstance(item, Box):
                if item.x + item.size[0] / 2 < most_left_distance:
                    most_left_distance = item.x + item.size[0] / 2
                if item.x + item.size[0] / 2 > most_right_distance:
                    most_right_distance = item.x + item.size[0] / 2
            if isinstance(item, Spider):
                if item.x < most_left_distance:
                    most_left_distance = item.x
                if item.x > most_right_distance:
                    most_right_distance = item.x

        for item in items:
            if isinstance(item, Box):
                if item.x + item.size[0] / 2 == most_left_distance:
                    most_left.append(item)
                if item.x + item.size[0] / 2 == most_right_distance:
                    most_right.append(item)
            if isinstance(item, Spider):
                if item.x == most_left_distance:
                    most_left.append(item)
                if item.x == most_right_distance:
                    most_right.append(item)

        most_left.sort(key=lambda obj: obj.y)
        most_right.sort(key=lambda obj: obj.y)

        left_connections = []
        right_connections = []

        for item in most_left:
            if isinstance(item, Box):
                for connection in item.connections:
                    if connection.side == "left" and connection.has_wire is False:
                        left_connections.append(connection)
            if isinstance(item, Spider):
                left_connections.append(item)
        for item in most_right:
            if isinstance(item, Box):
                for connection in item.connections:
                    if connection.side == "right" and connection.has_wire is False:
                        right_connections.append(connection)
            if isinstance(item, Spider):
                right_connections.append(item)

        return left_connections, right_connections

    def connect_extra_wires(self, copied_connections, connections, connected_amount):
        multiple_connections = []
        for connection in copied_connections:
            if connection.side == "spider":
                multiple_connections.append(connection)
        if len(copied_connections) >= len(connections) - connected_amount:
            for i in range(len(connections) - connected_amount):
                self.canvas.start_wire_from_connection(connections[i + connected_amount])
                self.canvas.end_wire_to_connection(copied_connections[i])
        else:
            for i in range(len(copied_connections)):
                self.canvas.start_wire_from_connection(connections[i + connected_amount])
                self.canvas.end_wire_to_connection(copied_connections[i])
            if len(multiple_connections) > 0:
                for i in range(len(connections) - connected_amount - len(copied_connections)):
                    self.canvas.start_wire_from_connection(connections[i + connected_amount + len(copied_connections)])
                    self.canvas.end_wire_to_connection(multiple_connections[i % len(multiple_connections)])

    def copy_selected_wires(self, connection_list, canvas=None):
        if canvas is None:
            canvas = self.canvas
        for wire in canvas.wires:
            connection = None
            if wire.start_connection in connection_list and wire.end_connection in connection_list:
                self.copied_wire_list.append({
                    'wire': "Wire",
                    'start_connection': None,
                    'end_connection': None,
                    'original_start_connection': copy.deepcopy(wire.start_connection.id),
                    'original_start_index': copy.deepcopy(wire.start_connection.index),
                    'original_start_side': copy.deepcopy(wire.start_connection.side),
                    'original_end_connection': copy.deepcopy(wire.end_connection.id),
                    'original_end_index': copy.deepcopy(wire.end_connection.index),
                    'original_end_side': copy.deepcopy(wire.end_connection.side)
                })
            elif wire.start_connection in connection_list and wire.end_connection not in connection_list:
                connection = wire.start_connection
            elif wire.start_connection not in connection_list and wire.end_connection in connection_list:
                connection = wire.end_connection
            if connection:
                if connection.side == "left":
                    self.add_copied_wire(connection, True)
                elif connection.side == "right":
                    self.add_copied_wire(connection, False)
                elif connection.side == "spider":
                    is_left = None
                    if wire.start_connection in connection_list:
                        if wire.end_connection.side == "left":
                            is_left = False
                        elif wire.end_connection.side == "right":
                            is_left = True
                        else:
                            is_left = connection.location[0] > wire.end_connection.location[0]
                    if wire.end_connection in connection_list:
                        if wire.start_connection.side == "left":
                            is_left = False
                        elif wire.start_connection.side == "right":
                            is_left = True
                        else:
                            is_left = connection.location[0] > wire.start_connection.location[0]
                    if is_left is not None:
                        self.add_copied_wire(connection, is_left)

    def add_edge_wires(self, pasted_items):
        left_connections, right_connections = self.find_side_connections()
        self.delete_selected_items()
        self.reconnect_wires(self.copied_left_wires, left_connections)
        self.reconnect_wires(self.copied_right_wires, right_connections)
        left_copied_connections, right_copied_connections = self.find_edge_items(pasted_items)
        if len(left_connections) > len(self.copied_left_wires):
            self.connect_extra_wires(left_copied_connections, left_connections, len(self.copied_left_wires))
        if len(right_connections) > len(self.copied_right_wires):
            self.connect_extra_wires(right_copied_connections, right_connections, len(self.copied_right_wires))
