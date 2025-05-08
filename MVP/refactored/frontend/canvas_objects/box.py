import json
import re
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import simpledialog

from MVP.refactored.frontend.canvas_objects.connection import Connection
from MVP.refactored.frontend.windows.code_editor import CodeEditor
from constants import *


class Box:
    def __init__(self, canvas, x, y, receiver, size=(60, 60), id_=None, shape="rectangle"):
        self.shape = shape
        self.canvas = canvas
        x, y = self.canvas.canvasx(x), self.canvas.canvasy(y)
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.size = size
        self.x_dif = 0
        self.y_dif = 0
        self.connections: list[Connection] = []
        self.left_connections = 0
        self.right_connections = 0
        self.label = None
        self.label_text = ""
        self.overlay = None
        self.overlay_text = ""
        self.wires = []
        if not id_:
            self.id = id(self)
        else:
            self.id = id_
        self.node = None
        self.context_menu = tk.Menu(self.canvas, tearoff=0)
        self.rect = self.create_rect()

        self.resize_handle = self.canvas.create_rectangle(self.x + self.size[0] - 10, self.y + self.size[1] - 10,
                                                          self.x + self.size[0], self.y + self.size[1],
                                                          outline="black", fill="black")
        self.locked = False
        self.bind_events()
        self.sub_diagram = None
        self.receiver = receiver
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("box_add", generator_id=self.id)
            if self.canvas.diagram_source_box:
                self.receiver.receiver_callback("sub_box", generator_id=self.id,
                                                connection_id=self.canvas.diagram_source_box.id)

        self.is_snapped = False

        self.collision_ids = [self.rect, self.resize_handle]
        self.neurons = 32
        self.activation = 'nn.ReLU'
        self.inputs = 10
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']
        self.outputs = 1
        self.dropout = 0.1
        self.out_channels = 16
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0
        self.pool_type = 'max'
        self.num_layers = 1
        self.bidirectional = False
        self.non_linearity = 'tanh'
        self.batch_first = True
        self.seq_to_seq = False

        self.num_heads = 8
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3

        self.category = None

    def set_id(self, id_):
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("box_swap_id", generator_id=self.id, connection_id=id_)
            if self.canvas.diagram_source_box:
                self.receiver.receiver_callback("sub_box", generator_id=self.id,
                                                connection_id=self.canvas.diagram_source_box.id)
        self.id = id_

    def bind_events(self):
        self.canvas.tag_bind(self.rect, '<Control-ButtonPress-1>', lambda event: self.on_control_press())
        self.canvas.tag_bind(self.rect, '<ButtonPress-1>', self.on_press)
        self.canvas.tag_bind(self.rect, '<B1-Motion>', self.on_drag)
        self.canvas.tag_bind(self.rect, '<ButtonPress-3>', self.show_context_menu)
        self.canvas.tag_bind(self.resize_handle, '<ButtonPress-1>', self.on_resize_press)
        self.canvas.tag_bind(self.resize_handle, '<B1-Motion>', self.on_resize_drag)
        self.canvas.tag_bind(self.resize_handle, '<Enter>', lambda _: self.canvas.on_hover(self))
        self.canvas.tag_bind(self.resize_handle, '<Leave>', lambda _: self.canvas.on_leave_hover())
        self.canvas.tag_bind(self.rect, '<Double-Button-1>', lambda _: self.handle_double_click())
        self.canvas.tag_bind(self.rect, '<Enter>', lambda _: self.canvas.on_hover(self))
        self.canvas.tag_bind(self.rect, '<Leave>', lambda _: self.canvas.on_leave_hover())

    def show_context_menu(self, event):
        self.close_menu()
        if self.context_menu is not None:
            try:
                self.context_menu.destroy()
                print("menu destroyed")
            except Exception as e:
                print("Error destroying menu:", e)
        self.context_menu = tk.Menu(self.canvas, tearoff=0)

        if not self.sub_diagram:
            self.context_menu.add_command(label="Add code", command=self.open_editor)
            if not self.label_text.strip():
                self.context_menu.entryconfig("Add code", state="disabled", label="Label needed to add code")

        if not self.locked and not self.sub_diagram:
            self.context_menu.add_command(label="Add Left Connection", command=self.add_left_connection)
            self.context_menu.add_command(label="Add Right Connection", command=self.add_right_connection)

            for circle in self.connections:
                self.context_menu.add_command(label=f"Remove {circle.side} connection nr {circle.index}",
                                              command=lambda bound_arg=circle: self.remove_connection(bound_arg))

            sub_menu = tk.Menu(self.context_menu, tearoff=0)
            self.context_menu.add_cascade(menu=sub_menu, label="Shape")
            sub_menu.add_command(label="Rectangle", command=lambda shape="rectangle": self.change_shape(shape))
            sub_menu.add_command(label="Triangle", command=lambda shape="triangle": self.change_shape(shape))

        if self.locked:
            self.context_menu.add_command(label="Unlock Box", command=self.unlock_box)
        if not self.locked:
            self.context_menu.add_command(label="Edit label", command=self.edit_label)
            self.context_menu.add_command(label="Edit Sub-Diagram", command=self.edit_sub_diagram)
            self.context_menu.add_command(label="Unfold sub-diagram", command=self.unfold)
            self.context_menu.add_command(label="Lock Box", command=self.lock_box)

            if "ffn output layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_output_layer_editor)
            elif "ffn hidden layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_middle_layer_editor)
            elif "ffn dropout" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_dropout_editor)

            if "cnn conv layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_conv_layer_editor)
            elif "cnn pool" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_pool_editor)
            elif "cnn dropout2d" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_dropout_editor)
            elif "cnn dense layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_middle_layer_editor)
            elif "cnn dropout" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_dropout_editor)
            elif "cnn output layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_middle_layer_editor)

            if "rnn input layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_rnn_input_layer_editor)
            elif "rnn lstm layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_rnn_layer_editor)
            elif "rnn gru layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_rnn_layer_editor)
            elif "rnn dropout" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_dropout_editor)
            elif "rnn simple layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_simple_rnn_layer_editor)
            elif "rnn output layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_middle_layer_editor)

            if "input dense layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_input_layer_editor)
            elif "output dense layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_output_layer_editor)
            elif "dense layer" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_middle_layer_editor)
            elif "transformer 2" in self.label_text:
                self.context_menu.add_command(label="Edit Properties", command=self.open_transformer_layer_editor)

        self.context_menu.add_command(label="Save Box to Menu", command=self.save_box_to_menu)
        self.context_menu.add_command(label="Save AI Box to Menu", command=self.save_ai_box_to_menu)
        if self.sub_diagram:
            self.context_menu.add_command(label="Delete Box", command=lambda: self.delete_box(action="sub_diagram"))
        else:
            self.context_menu.add_command(label="Delete Box", command=self.delete_box)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Cancel")
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def unfold(self):
        if not self.sub_diagram:
            return
        event = tk.Event()
        event.x = self.x + self.size[0] / 2
        event.y = self.y + self.size[1] / 2
        self.sub_diagram.select_all()
        self.canvas.selector.copy_selected_items(canvas=self.sub_diagram)
        self.on_press(event)
        self.canvas.paste_copied_items(event)

    def open_editor(self):
        CodeEditor(self.canvas.main_diagram, box=self)

    def save_box_to_menu(self):
        if not self.label_text:
            self.edit_label()
        if not self.label_text:
            return
        self.canvas.main_diagram.save_box_to_diagram_menu(self)

    def save_ai_box_to_menu(self):
        if not self.label_text:
            self.edit_label()
        if not self.label_text:
            return
        self.category = "ai"
        self.canvas.main_diagram.save_box_to_ai_diagram_menu(self)

    def handle_double_click(self):
        if self.sub_diagram:
            self.canvas.main_diagram.switch_canvas(self.sub_diagram)
        else:
            self.set_inputs_outputs()

    def set_inputs_outputs(self):
        if self.locked:
            return
        # ask for inputs amount
        inputs = simpledialog.askstring(title="Inputs (left connections)", prompt="Enter amount")
        if inputs and not inputs.isdigit():
            while True:
                inputs = simpledialog.askstring(title="Inputs (left connections)",
                                                prompt="Enter amount, must be integer!")
                if inputs:
                    if inputs.isdigit():
                        break
                else:
                    break

        # ask for outputs amount
        outputs = simpledialog.askstring(title="Outputs (right connections)", prompt="Enter amount")
        if outputs and not outputs.isdigit():
            while True:
                outputs = simpledialog.askstring(title="Outputs (right connections)",
                                                 prompt="Enter amount, must be integer!")
                if outputs:
                    if outputs.isdigit():
                        break
                else:
                    break
        # select connections to remove

        to_be_removed = []
        for c in self.connections:
            if c.side == "right" and outputs:
                to_be_removed.append(c)
            if c.side == "left" and inputs:
                to_be_removed.append(c)

        # remove selected connectionsS
        for c in to_be_removed:
            c.delete()
            self.remove_connection(c)
            self.update_connections()
            self.update_wires()

        # add new connections
        if not self.canvas.search:
            self.receiver.receiver_callback("box_remove_connection_all", generator_id=self.id)
        if outputs:
            for _ in range(int(outputs)):
                self.add_right_connection()
        if inputs:
            for _ in range(int(inputs)):
                self.add_left_connection()

    def edit_sub_diagram(self, save_to_canvasses=True, add_boxes=True, switch=True):
        from MVP.refactored.frontend.components.custom_canvas import CustomCanvas
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("compound", generator_id=self.id)
        if not self.sub_diagram:
            self.sub_diagram = CustomCanvas(self.canvas.main_diagram, self, self.receiver, self.canvas.main_diagram,
                                            self.canvas, add_boxes, self.id, highlightthickness=0)
            self.canvas.itemconfig(self.rect, fill="#dfecf2")
            if save_to_canvasses:
                name = self.label_text
                if not name:
                    name = str(self.sub_diagram.id)[-6:]
                    self.set_label(name)
                self.sub_diagram.set_name(name)
                self.canvas.main_diagram.add_canvas(self.sub_diagram)
                self.canvas.main_diagram.change_canvas_name(self.sub_diagram)
                if switch:
                    self.canvas.main_diagram.switch_canvas(self.sub_diagram)

            return self.sub_diagram
        else:
            if switch:
                self.canvas.main_diagram.switch_canvas(self.sub_diagram)
            return self.sub_diagram

    def close_menu(self):
        if self.context_menu:
            self.context_menu.destroy()

    # MOVING, CLICKING ETC.
    def on_press(self, event):
        event.x, event.y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        for item in self.canvas.selector.selected_items:
            item.deselect()
        self.canvas.selector.selected_boxes.clear()
        self.canvas.selector.selected_spiders.clear()
        self.canvas.selector.selected_wires.clear()
        self.canvas.selector.selected_items.clear()
        self.select()
        self.canvas.selector.selected_items.append(self)
        self.start_x = event.x
        self.start_y = event.y
        self.x_dif = event.x - self.x
        self.y_dif = event.y - self.y

    def on_control_press(self):
        if self in self.canvas.selector.selected_items:
            self.canvas.selector.selected_items.remove(self)
            self.deselect()
        else:
            self.select()
            self.canvas.selector.selected_items.append(self)
        self.canvas.selector.select_wires_between_selected_items()

    def on_drag(self, event):
        if event.state & 0x4:
            return
        event.x, event.y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        self.start_x = event.x
        self.start_y = event.y

        go_to_x = event.x - self.x_dif
        go_to_y = event.y - self.y_dif

        # snapping into place
        found = False
        for box in self.canvas.boxes:
            if box == self:
                continue

            if abs(box.x + box.size[0] / 2 - (go_to_x + self.size[0] / 2)) < box.size[0] / 2 + self.size[0] / 2:
                go_to_x = box.x + box.size[0] / 2 - +self.size[0] / 2

                found = True
        for spider in self.canvas.spiders:

            if abs(spider.location[0] - (go_to_x + self.size[0] / 2)) < self.size[0] / 2 + spider.r:
                go_to_x = spider.x - +self.size[0] / 2

                found = True

        if found:
            collision = self.find_collisions(go_to_x, go_to_y)

            if len(collision) != 0:
                if self.is_snapped:
                    return

                jump_size = 10
                counter = 0
                while collision:
                    if counter % 2 == 0:
                        go_to_y += counter * jump_size
                    else:
                        go_to_y -= counter * jump_size

                    collision = self.find_collisions(go_to_x, go_to_y)

                    counter += 1

        self.is_snapped = found

        self.move(go_to_x, go_to_y)
        self.move_label()

    def get_self_collision_ids(self):
        self.collision_ids = [self.rect, self.resize_handle]
        if self.label:
            self.collision_ids.append(self.label)
        for connection in self.connections:
            self.collision_ids.append(connection.circle)

    def find_collisions(self, go_to_x, go_to_y):
        self.get_self_collision_ids()
        collision = self.canvas.find_overlapping(go_to_x, go_to_y, go_to_x + self.size[0], go_to_y + self.size[1])
        collision = list(collision)
        for index in self.collision_ids:
            if index in collision:
                collision.remove(index)
        for wire in self.canvas.wires:
            tag = wire.line
            if tag in collision:
                collision.remove(tag)
        return collision

    def on_resize_scroll(self, event):
        if event.delta == 120:
            multiplier = 1
        else:
            multiplier = -1
        if multiplier == -1:
            if 20 > min(self.size):
                return
        old_size = self.size
        self.size = (self.size[0] + 5 * multiplier, self.size[1] + 5 * multiplier)
        if self.find_collisions(self.x, self.y):
            self.size = old_size
            return
        self.update_size(self.size[0] + 5 * multiplier, self.size[1] + 5 * multiplier)
        self.move_label()

    def on_resize_drag(self, event):
        event.x, event.y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        resize_x = self.x + self.size[0] - 10
        resize_y = self.y + self.size[1] - 10
        dx = event.x - self.start_x
        dy = event.y - self.start_y

        if dx > 0 and not resize_x <= event.x:
            dx = 0

        if dy > 0 and not resize_y <= event.y:
            dy = 0

        self.start_x = event.x
        self.start_y = event.y
        new_size_x = max(20, self.size[0] + dx)
        new_size_y = max(20, self.size[1] + dy)
        self.update_size(new_size_x, new_size_y)
        self.move_label()

    def resize_by_connections(self):
        # TODO resize by label too if needed
        nr_cs = max([c.index for c in self.connections] + [0])
        height = max([50 * nr_cs, 50])
        if self.size[1] < height:
            self.update_size(self.size[0], height)
            self.move_label()

    def move_label(self):
        if not self.label:
            return

        layer_labels = {"input dense layer", "dense layer", "output dense layer"}
        if self.label_text in layer_labels:
            self.canvas.coords(
                self.label,
                self.x + self.size[0] / 2,
                self.y + 10  # 10 pixels from the top edge of the box
            )

            if self.overlay:
                label_bbox = self.canvas.bbox(self.label)
                if label_bbox:
                    x_center = (label_bbox[0] + label_bbox[2]) / 2
                    label_bottom = label_bbox[3]

                    overlay_bbox = self.canvas.bbox(self.overlay)
                    if overlay_bbox:
                        overlay_height = overlay_bbox[3] - overlay_bbox[1]
                        overlay_center_y = label_bottom + 5 + overlay_height / 2
                        self.canvas.coords(self.overlay, x_center, overlay_center_y)
                    else:
                        self.canvas.coords(self.overlay, x_center, self.y + 60)
        else:
            self.canvas.coords(
                self.label,
                self.x + self.size[0] / 2,
                # self.y + self.size[1] / 2
                self.y + 10
            )

    def bind_event_label(self):
        self.canvas.tag_bind(self.label, '<B1-Motion>', self.on_drag)
        self.canvas.tag_bind(self.label, '<ButtonPress-3>', self.show_context_menu)
        self.canvas.tag_bind(self.label, '<Double-Button-1>', lambda _: self.handle_double_click())
        self.canvas.tag_bind(self.label, '<Control-ButtonPress-1>', lambda event: self.on_control_press())
        self.canvas.tag_bind(self.label, '<ButtonPress-1>', self.on_press)
        self.canvas.tag_bind(self.label, '<Enter>', lambda _: self.canvas.on_hover(self))
        self.canvas.tag_bind(self.label, '<Leave>', lambda _: self.canvas.on_leave_hover())

    def edit_label(self, new_label=None):
        if new_label is None:
            text = simpledialog.askstring("Input", "Enter label:", initialvalue=self.label_text)
            if text is not None:
                self.label_text = text
            if os.stat(FUNCTIONS_CONF).st_size != 0:
                with open(FUNCTIONS_CONF, "r") as file:
                    data = json.load(file)
                    for label, code in data.items():
                        if label == self.label_text:
                            if messagebox.askokcancel("Confirmation",
                                                      "A box with this label already exists."
                                                      " Do you want to use the existing box?"):
                                self.update_io()
                            else:
                                return self.edit_label()
        else:
            self.label_text = new_label

        self.change_label()

        if self.label_text:
            if self.sub_diagram:
                self.sub_diagram.set_name(self.label_text)
                self.canvas.main_diagram.change_canvas_name(self.sub_diagram)

        self.bind_event_label()

    def open_rnn_input_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit RNN Input Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Batch First:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        bf_options = ['False', 'True']
        curr_bf = str(getattr(self, 'batch_first', True))
        if curr_bf not in bf_options:
            bf_options.insert(0, curr_bf)
        bf_var = tk.StringVar(value=curr_bf)
        bf_combo = ttk.Combobox(editor, textvariable=bf_var, values=bf_options, state="readonly")
        bf_combo.grid(row=0, column=1, padx=5, pady=5)
        try:
            bf_combo.current(bf_options.index(curr_bf))
        except ValueError:
            bf_combo.current(0)

        tk.Label(editor, text="Seq to Seq:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        sts_options = ['False', 'True']
        curr_sts = str(getattr(self, 'seq_to_seq', False))
        if curr_sts not in sts_options:
            sts_options.insert(0, curr_sts)
        sts_var = tk.StringVar(value=curr_sts)
        sts_combo = ttk.Combobox(editor, textvariable=sts_var, values=sts_options, state="readonly")
        sts_combo.grid(row=1, column=1, padx=5, pady=5)
        try:
            sts_combo.current(sts_options.index(curr_sts))
        except ValueError:
            sts_combo.current(0)

        def save_input_properties():
            new_bf = True if bf_var.get() == 'True' else False
            new_sts = True if sts_var.get() == 'True' else False
            self.batch_first = new_bf
            self.seq_to_seq = new_sts
            self.change_label()
            editor.destroy()

        tk.Button(editor, text="Save", command=save_input_properties).grid(row=2, column=0, columnspan=2, pady=10)

    def open_simple_rnn_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit RNN Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Neurons:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        neurons_var = tk.StringVar(value=str(getattr(self, 'neurons', '')))
        tk.Entry(editor, textvariable=neurons_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Num Layers:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        layers_var = tk.StringVar(value=str(getattr(self, 'num_layers', 1)))
        tk.Entry(editor, textvariable=layers_var).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(editor, text="Non Linearity:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        nonlin_options = ['tanh', 'relu']
        curr_nonlin = getattr(self, 'non_linearity', 'tanh')
        if curr_nonlin not in nonlin_options:
            nonlin_options.insert(0, curr_nonlin)
        nonlin_var = tk.StringVar(value=curr_nonlin)
        nonlin_combo = ttk.Combobox(editor, textvariable=nonlin_var, values=nonlin_options, state="readonly")
        nonlin_combo.grid(row=2, column=1, padx=5, pady=5)
        try:
            nonlin_combo.current(nonlin_options.index(curr_nonlin))
        except ValueError:
            nonlin_combo.current(0)

        tk.Label(editor, text="Bidirectional:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        bidi_options = ['False', 'True']
        curr_bidi = str(getattr(self, 'bidirectional', False))
        if curr_bidi not in bidi_options:
            bidi_options.insert(0, curr_bidi)
        bidi_var = tk.StringVar(value=curr_bidi)
        bidi_combo = ttk.Combobox(editor, textvariable=bidi_var, values=bidi_options, state="readonly")
        bidi_combo.grid(row=3, column=1, padx=5, pady=5)
        try:
            bidi_combo.current(bidi_options.index(curr_bidi))
        except ValueError:
            bidi_combo.current(0)

        tk.Label(editor, text="Dropout:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        dropout_var = tk.StringVar(value=str(getattr(self, 'dropout', 0.0)))
        tk.Entry(editor, textvariable=dropout_var).grid(row=4, column=1, padx=5, pady=5)

        def save_simple_rnn_properties():

            try:
                new_neurons = int(neurons_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Hidden Dim must be an integer.")
                return

            try:
                new_layers = int(layers_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Num Layers must be an integer.")
                return

            new_nonlin = nonlin_var.get().strip()
            if new_nonlin not in ['tanh', 'relu']:
                messagebox.showerror("Invalid Value", "Nonlinearity must be 'tanh' or 'relu'.")
                return

            new_bidi = True if bidi_var.get() == 'True' else False

            try:
                new_dropout = float(dropout_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Dropout must be a float.")
                return

            self.neurons = new_neurons
            self.num_layers = new_layers
            self.non_linearity = new_nonlin
            self.bidirectional = new_bidi
            self.dropout = new_dropout
            self.change_label()
            editor.destroy()

        tk.Button(editor, text="Save", command=save_simple_rnn_properties).grid(row=5, column=0, columnspan=2, pady=10)

    def open_rnn_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit RNN Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Neurons:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        neurons_var = tk.StringVar(value=str(getattr(self, 'neurons', '')))
        tk.Entry(editor, textvariable=neurons_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Num Layers:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        layers_var = tk.StringVar(value=str(getattr(self, 'num_layers', 1)))
        tk.Entry(editor, textvariable=layers_var).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(editor, text="Bidirectional:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        bidi_options = ['False', 'True']
        curr_bidi = str(getattr(self, 'bidirectional', False))
        if curr_bidi not in bidi_options:
            bidi_options.insert(0, curr_bidi)
        bidi_var = tk.StringVar(value=curr_bidi)
        ttk.Combobox(editor, textvariable=bidi_var, values=bidi_options, state="readonly").grid(row=2, column=1, padx=5, pady=5)
        try:
            editor.children[list(editor.children.keys())[-1]].current(bidi_options.index(curr_bidi))
        except Exception:
            editor.children[list(editor.children.keys())[-1]].current(0)

        tk.Label(editor, text="Dropout:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        dropout_var = tk.StringVar(value=str(getattr(self, 'dropout', 0.0)))
        tk.Entry(editor, textvariable=dropout_var).grid(row=3, column=1, padx=5, pady=5)

        def save_rnn_properties():
            try:
                new_neurons = int(neurons_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Hidden Dim must be an integer.")
                return
            try:
                new_layers = int(layers_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Num Layers must be an integer.")
                return
            new_bidi = True if bidi_var.get()=='True' else False
            try:
                new_dropout = float(dropout_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Dropout must be a float.")
                return
            self.neurons = new_neurons
            self.num_layers = new_layers
            self.bidirectional = new_bidi
            self.dropout = new_dropout
            self.change_label()
            editor.destroy()

        tk.Button(editor, text="Save", command=save_rnn_properties).grid(row=4, column=0, columnspan=2, pady=10)

    def open_pool_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Pooling Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Kernel Size:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ksize_var = tk.StringVar(value=str(getattr(self, 'kernel_size', '')))
        tk.Entry(editor, textvariable=ksize_var).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Stride:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        stride_var = tk.StringVar(value=str(getattr(self, 'stride', '')))
        tk.Entry(editor, textvariable=stride_var).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(editor, text="Pool Type:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        types = ['max', 'avg']
        current_type = getattr(self, 'pool_type', 'max')
        if current_type not in types:
            types.insert(0, current_type)
        type_var = tk.StringVar(value=current_type)
        type_combo = ttk.Combobox(editor, textvariable=type_var, values=types, state="readonly")
        type_combo.grid(row=2, column=1, padx=5, pady=5)
        try:
            type_combo.current(types.index(current_type))
        except ValueError:
            type_combo.current(0)

        def save_pool_properties():
            def parse_val(text):
                t = text.strip()
                if ',' in t:
                    p = [x.strip() for x in t.split(',')]
                    if len(p) != 2:
                        raise ValueError
                    return (int(p[0]), int(p[1]))
                return int(t)
            try:
                new_ks = parse_val(ksize_var.get())
                new_stride = parse_val(stride_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Value",
                    "Kernel Size and Stride must be int or two ints separated by comma."
                )
                return
            new_type = type_var.get().strip()
            if new_type not in ['max', 'avg']:
                messagebox.showerror("Invalid Value", "Pool type must be 'max' or 'avg'.")
                return
            self.kernel_size = new_ks
            self.stride = new_stride
            self.pool_type = new_type
            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_pool_properties)
        save_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def open_conv_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Convolution Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Out Channels:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        channels_var = tk.StringVar(value=str(getattr(self, 'out_channels', '')))
        channels_entry = tk.Entry(editor, textvariable=channels_var)
        channels_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Kernel Size:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ksize_var = tk.StringVar(value=str(getattr(self, 'kernel_size', '')))
        ksize_entry = tk.Entry(editor, textvariable=ksize_var)
        ksize_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(editor, text="Stride:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        stride_var = tk.StringVar(value=str(getattr(self, 'stride', '')))
        stride_entry = tk.Entry(editor, textvariable=stride_var)
        stride_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(editor, text="Padding:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        pad_var = tk.StringVar(value=str(getattr(self, 'padding', '')))
        pad_entry = tk.Entry(editor, textvariable=pad_var)
        pad_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(editor, text="Activation:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        activations = [
            "nn.ReLU",
            "nn.Sigmoid",
            "nn.Tanh",
            "nn.LeakyReLU",
            "nn.ELU",
            "nn.GELU",
            "nn.Softmax",
        ]
        current = str(self.activation)
        if current not in activations:
            activations.insert(0, current)
        activation_var = tk.StringVar(value=current)
        activation_combo = ttk.Combobox(
            editor,
            textvariable=activation_var,
            values=activations,
            state="readonly"
        )
        activation_combo.grid(row=4, column=1, padx=5, pady=5)
        # safe preselect
        try:
            activation_combo.current(activations.index(current))
        except ValueError:
            activation_combo.current(0)

        def save_properties():
            try:
                new_channels = int(channels_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Out Channels must be an integer.")
                return

            def parse_val(text):
                t = text.strip()
                if ',' in t:
                    parts = [p.strip() for p in t.split(',')]
                    if len(parts) != 2:
                        raise ValueError
                    return (int(parts[0]), int(parts[1]))
                return int(t)

            try:
                new_ks = parse_val(ksize_var.get())
                new_stride = parse_val(stride_var.get())
                new_pad = parse_val(pad_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid Value",
                    "Kernel Size, Stride, and Padding must be int or two ints separated by comma."
                )
                return
            act_name = activation_var.get().strip()
            # act_map = {
            #     'nn.ReLU': nn.ReLU,
            #     'nn.Sigmoid': nn.Sigmoid,
            #     'nn.Tanh': nn.Tanh,
            #     'nn.LeakyReLU': nn.LeakyReLU,
            #     'nn.ELU': nn.ELU,
            #     'nn.GELU': nn.GELU,
            #     'nn.Softmax': nn.Softmax,
            # }
            # act_cls = act_map.get(act_name)
            # if act_cls is None:
            #     messagebox.showerror("Invalid Value", "Unknown activation function.")
            #     return
            self.out_channels = new_channels
            self.kernel_size = new_ks
            self.stride = new_stride
            self.padding = new_pad
            self.activation = act_name
            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_properties)
        save_btn.grid(row=5, column=0, columnspan=2, pady=10)

    def open_dropout_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Dropout")
        editor.grab_set()

        tk.Label(editor, text="Dropout:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        dropout_var = tk.StringVar(value=str(self.dropout))
        dropout_entry = tk.Entry(editor, textvariable=dropout_var)
        dropout_entry.grid(row=0, column=1, padx=5, pady=5)

        def save_properties():
            try:
                new_dropout = float(dropout_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Dropout must be an number.")
                return

            self.dropout = new_dropout

            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_properties)
        save_btn.grid(row=1, column=0, columnspan=2, pady=10)

    def open_middle_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Middle Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Neurons:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        neurons_var = tk.StringVar(value=str(self.neurons))
        neurons_entry = tk.Entry(editor, textvariable=neurons_var)
        neurons_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Activation:").grid(row=1, column=0, padx=5, pady=5, sticky="e")

        activations = [
            "None",
            "nn.ReLU",
            "nn.Sigmoid",
            "nn.Tanh",
            "nn.LeakyReLU",
            "nn.ELU",
            "nn.GELU",
            "nn.Softmax",
            "nn.LogSoftmax",
        ]

        current = str(self.activation)
        if current not in activations:
            activations.insert(0, current)

        activation_var = tk.StringVar(value=self.activation)
        activation_combo = ttk.Combobox(
            editor,
            textvariable=activation_var,
            values=activations,
            state="readonly"
        )
        activation_combo.grid(row=1, column=1, padx=5, pady=5)
        activation_combo.current(activations.index(self.activation))

        def save_properties():
            try:
                new_neurons = int(neurons_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Neurons must be an integer.")
                return

            new_activation = activation_var.get()
            if not new_activation:
                messagebox.showerror("Invalid Value", "Please select an activation function.")
                return

            self.neurons = new_neurons
            self.activation = new_activation

            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_properties)
        save_btn.grid(row=2, column=0, columnspan=2, pady=10)

    def open_input_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Input Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Inputs:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        inputs_var = tk.StringVar(value=str(self.inputs))
        inputs_entry = tk.Entry(editor, textvariable=inputs_var)
        inputs_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Neurons:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        neurons_var = tk.StringVar(value=str(self.neurons))
        neurons_entry = tk.Entry(editor, textvariable=neurons_var)
        neurons_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(editor, text="Activation:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        activation_var = tk.StringVar(value=self.activation)
        activation_entry = tk.Entry(editor, textvariable=activation_var)
        activation_entry.grid(row=2, column=1, padx=5, pady=5)

        def save_inputs():
            try:
                new_inputs = int(inputs_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Inputs must be an integer.")
                return
            try:
                new_neurons = int(neurons_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Neurons must be an integer.")
                return
            new_activation = activation_var.get().strip()
            if not new_activation:
                messagebox.showerror("Invalid Value", "Activation function cannot be empty.")
                return

            self.inputs = new_inputs
            self.neurons = new_neurons
            self.activation = new_activation

            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_inputs)
        save_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def open_output_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Output Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Output Neurons:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        output_neurons_var = tk.StringVar(value=str(self.outputs))
        output_neurons_entry = tk.Entry(editor, textvariable=output_neurons_var)
        output_neurons_entry.grid(row=0, column=1, padx=5, pady=5)

        def save_inputs():
            try:
                new_output_neurons = int(output_neurons_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Neurons must be an integer.")
                return

            self.outputs = new_output_neurons

            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_inputs)
        save_btn.grid(row=1, column=0, columnspan=2, pady=10)

    def open_transformer_layer_editor(self):
        editor = tk.Toplevel(self.canvas)
        editor.title("Edit Transformer Layer Properties")
        editor.grab_set()

        tk.Label(editor, text="Number of Heads:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        num_heads_var = tk.StringVar(value=str(self.num_heads))
        num_heads_entry = tk.Entry(editor, textvariable=num_heads_var)
        num_heads_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(editor, text="Number of Encoder Layers:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        encoder_layers_var = tk.StringVar(value=str(self.num_encoder_layers))
        encoder_layers_entry = tk.Entry(editor, textvariable=encoder_layers_var)
        encoder_layers_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(editor, text="Number of Decoder Layers:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        decoder_layers_var = tk.StringVar(value=str(self.num_decoder_layers))
        decoder_layers_entry = tk.Entry(editor, textvariable=decoder_layers_var)
        decoder_layers_entry.grid(row=2, column=1, padx=5, pady=5)

        def save_inputs():
            try:
                new_num_heads = int(num_heads_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Number of heads must be an integer.")
                return
            try:
                new_encoder_layers = int(encoder_layers_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Number of encoder layers must be an integer.")
                return
            try:
                new_decoder_layers = int(decoder_layers_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", "Number of decoder layers must be an integer.")
                return

            self.num_heads = new_num_heads
            self.num_encoder_layers = new_encoder_layers
            self.num_decoder_layers = new_decoder_layers

            self.change_label()
            editor.destroy()

        save_btn = tk.Button(editor, text="Save", command=save_inputs)
        save_btn.grid(row=3, column=0, columnspan=2, pady=10)

    def change_label(self):
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("box_add_operator", generator_id=self.id, operator=self.label_text)

        layer_labels = {"input dense layer", "dense layer", "output dense layer"}
        if self.label_text in layer_labels:
            if not self.label:
                self.label = self.canvas.create_text(
                    self.x + self.size[0] / 2,
                    self.y + self.size[1] / 2 + 10,
                    text=self.label_text, fill="black", font=('Helvetica', 14)
                )
                self.collision_ids.append(self.label)
            else:
                self.canvas.itemconfig(self.label, text=self.label_text, font=('Helvetica', 14))

            self.overlay_text = self.get_overlay_text()
            if self.overlay is None:
                self.overlay = self.canvas.create_text(
                    self.x + self.size[0] / 2,
                    self.y + 60,
                    text=self.overlay_text,
                    fill="black",
                    font=('Helvetica', 10)
                )
                self.collision_ids.append(self.overlay)
            else:
                self.canvas.itemconfig(self.overlay, text=self.overlay_text, font=('Helvetica', 10))
            self.move_label()
        else:

            if not self.label:
                self.label = self.canvas.create_text((self.x + self.size[0] / 2,
                                                      # self.y + self.size[1] / 2
                                                      self.y + 10
                                                      ),
                                                     text=self.label_text, fill="black", font=('Helvetica', 14))
                self.collision_ids.append(self.label)
            else:
                self.canvas.itemconfig(self.label, text=self.label_text)
            if self.overlay is not None:
                self.canvas.delete(self.overlay)
                self.overlay = None
                self.overlay_text = ""

    def get_overlay_text(self):
        if self.label_text == "dense layer":
            return f"Neurons: {self.neurons}\nActivation: {self.activation}"
        elif self.label_text == "input dense layer":
            return f"Input Neurons: {self.inputs}\nNeurons: {self.neurons}\nActivation: {self.activation}"
        elif self.label_text == "output dense layer":
            return f"Output Neurons: {self.outputs}\nActivation: {self.activation}\nOptimizer: {self.optimizer}\n" \
                   f"Loss Function: {self.loss}\nMetrics: {self.metrics}"

    def set_label(self, new_label):
        self.label_text = new_label
        self.change_label()
        self.bind_event_label()

    def on_resize_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def move(self, new_x, new_y):
        new_x = round(new_x, 4)
        new_y = round(new_y, 4)
        is_bad = False
        for connection in self.connections:
            if connection.has_wire and self.is_illegal_move(connection, new_x):
                is_bad = True
                break
        if is_bad:
            self.y = new_y
            self.update_position()
            self.update_connections()
            self.update_wires()
        else:
            self.x = new_x
            self.y = new_y
            self.update_position()
            self.update_connections()
            self.update_wires()

    def select(self):
        self.canvas.itemconfig(self.rect, outline="green")
        [c.select() for c in self.connections]

    def search_highlight_secondary(self):
        self.canvas.itemconfig(self.rect, outline="orange")
        [c.search_highlight_secondary() for c in self.connections]
        self.canvas.search_result_highlights.append(self)

    def search_highlight_primary(self):
        self.canvas.itemconfig(self.rect, outline="cyan")
        [c.search_highlight_primary() for c in self.connections]
        self.canvas.search_result_highlights.append(self)

    def deselect(self):
        self.canvas.itemconfig(self.rect, outline="black")
        [c.deselect() for c in self.connections]

    def lock_box(self):
        self.locked = True

    def unlock_box(self):
        self.locked = False

    # UPDATES
    def update_size(self, new_size_x, new_size_y):
        self.size = (new_size_x, new_size_y)
        self.update_position()
        self.update_connections()
        self.update_wires()

    def update_position(self):
        if self.shape == "rectangle":
            self.canvas.coords(self.rect, self.x, self.y, self.x + self.size[0], self.y + self.size[1])
        if self.shape == "triangle":
            self.canvas.coords(self.rect,
                               self.x + self.size[0], self.y + self.size[1] / 2,
                               self.x, self.y,
                               self.x, self.y + self.size[1])
        self.canvas.coords(self.resize_handle, self.x + self.size[0] - 10, self.y + self.size[1] - 10,
                           self.x + self.size[0], self.y + self.size[1])

    def update_connections(self):
        for c in self.connections:
            conn_x, conn_y = self.get_connection_coordinates(c.side, c.index)
            c.move_to((conn_x, conn_y))

    def update_wires(self):
        [wire.update() for wire in self.wires]

    def update_io(self):
        """Update inputs and outputs based on label and code."""
        with open(FUNCTIONS_CONF, "r") as file:
            data = json.load(file)
            for label, code in data.items():
                if label == self.label_text:
                    inputs_amount, outputs_amount = self.get_input_output_amount_off_code(code)
                    if inputs_amount > self.left_connections:
                        for i in range(inputs_amount - self.left_connections):
                            self.add_left_connection()
                    elif inputs_amount < self.left_connections:
                        for j in range(self.left_connections - inputs_amount):
                            for con in self.connections[::-1]:
                                if con.side == "left":
                                    self.remove_connection(con)
                                    break

                    if outputs_amount > self.right_connections:
                        for i in range(outputs_amount - self.right_connections):
                            self.add_right_connection()
                    elif outputs_amount < self.right_connections:
                        for i in range(self.right_connections - outputs_amount):
                            for con in self.connections[::-1]:
                                if con.side == "right":
                                    self.remove_connection(con)
                                    break

    # ADD TO/REMOVE FROM CANVAS
    def add_wire(self, wire):
        if wire not in self.wires:
            self.wires.append(wire)

    def add_left_connection(self, id_=None):
        i = self.get_new_left_index()
        conn_x, conn_y = self.get_connection_coordinates("left", i)
        connection = Connection(self, i, "left", (conn_x, conn_y), self.canvas, id_=id_)
        self.left_connections += 1
        self.connections.append(connection)
        self.collision_ids.append(connection.circle)

        self.update_connections()
        self.update_wires()
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("box_add_left", generator_id=self.id, connection_nr=i,
                                            connection_id=connection.id)

        self.resize_by_connections()
        return connection

    def add_right_connection(self, id_=None):
        i = self.get_new_right_index()
        conn_x, conn_y = self.get_connection_coordinates("right", i)
        connection = Connection(self, i, "right", (conn_x, conn_y), self.canvas, id_=id_)
        self.right_connections += 1
        self.connections.append(connection)
        self.collision_ids.append(connection.circle)

        self.update_connections()
        self.update_wires()
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("box_add_right", generator_id=self.id, connection_nr=i,
                                            connection_id=connection.id)
        self.resize_by_connections()
        return connection

    def remove_connection(self, circle):
        for c in self.connections:
            if c.index > circle.index and circle.side == c.side:
                c.lessen_index_by_one()
        if self.receiver.listener and not self.canvas.search:
            self.receiver.receiver_callback("box_remove_connection", generator_id=self.id, connection_nr=circle.index,
                                            generator_side=circle.side)
        if circle.side == "left":
            self.left_connections -= 1
        elif circle.side == "right":
            self.right_connections -= 1

        self.connections.remove(circle)
        self.collision_ids.remove(circle.circle)
        circle.delete()
        self.update_connections()
        self.update_wires()
        self.resize_by_connections()

    def delete_box(self, keep_sub_diagram=False, action=None):
        for c in self.connections:
            c.delete()

        self.canvas.delete(self.rect)
        self.canvas.delete(self.resize_handle)
        layer_labels = {"input dense layer", "dense layer", "output dense layer"}
        if self.label_text in layer_labels:
            self.canvas.delete(self.overlay)
        if self in self.canvas.boxes:
            self.canvas.boxes.remove(self)
        self.canvas.delete(self.label)
        if self.sub_diagram and not keep_sub_diagram:
            self.canvas.main_diagram.del_from_canvasses(self.sub_diagram)
        if self.receiver.listener and not self.canvas.search:
            if action != "sub_diagram":
                self.receiver.receiver_callback("box_delete", generator_id=self.id)

    # BOOLEANS
    def is_illegal_move(self, connection, new_x):
        wire = connection.wire
        if connection.side == "left":
            if connection == wire.start_connection:
                other_connection = wire.end_connection
            else:
                other_connection = wire.start_connection
            other_x = other_connection.location[0]
            if other_x + other_connection.width_between_boxes >= new_x:
                return True

        if connection.side == "right":
            if connection == wire.start_connection:
                other_connection = wire.end_connection
            else:
                other_connection = wire.start_connection

            other_x = other_connection.location[0]
            if other_x - other_connection.width_between_boxes <= new_x + self.size[0]:
                return True
        return False

    # HELPERS
    def get_connection_coordinates(self, side, index):
        if side == "left":
            i = self.get_new_left_index()
            return self.x, self.y + (index + 1) * self.size[1] / (i + 1)

        elif side == "right":
            i = self.get_new_right_index()
            return self.x + self.size[0], self.y + (index + 1) * self.size[1] / (i + 1)

    def get_new_left_index(self):
        if not self.left_connections > 0:
            return 0
        return max([c.index if c.side == "left" else 0 for c in self.connections]) + 1

    def get_new_right_index(self):
        if not self.right_connections > 0:
            return 0
        return max([c.index if c.side == "right" else 0 for c in self.connections]) + 1

    def create_rect(self):
        w, h = self.size
        if self.shape == "rectangle":
            return self.canvas.create_rectangle(self.x, self.y, self.x + w, self.y + h,
                                                outline="black", fill="white")
        if self.shape == "triangle":
            return self.canvas.create_polygon(self.x + w, self.y + h / 2, self.x, self.y,
                                              self.x, self.y + h, outline="black", fill="white")

    def change_shape(self, shape):
        if shape == "rectangle":
            new_box = self.canvas.add_box((self.x, self.y), self.size, shape="rectangle")
        elif shape == "triangle":
            new_box = self.canvas.add_box((self.x, self.y), self.size, shape="triangle")
        else:
            return
        self.canvas.copier.copy_box(self, new_box)
        self.delete_box()

    @staticmethod
    def get_input_output_amount_off_code(code):
        inputs = re.search(r"\((.*)\)", code).group(1)
        outputs = re.search(r"return (.*)\n*", code).group(1)
        inputs_amount = len(inputs.split(","))
        if outputs[0] == "(":
            outputs = outputs[1:-1]
        outputs_amount = len(outputs.strip().split(","))
        if not inputs:
            inputs_amount = 0
        if not outputs:
            outputs_amount = 0
        return inputs_amount, outputs_amount
