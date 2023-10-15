



import pyglet
import numpy as np
from pyglet.gl import *
import pyglet as pglt


class NeuralNetworkSize:
    def __init__(self) -> None:
        self.gap_layer = 0
        self.gap_neuron = 0
        self.layers = []
        self.layers_bp = []

        self.layer_num = 0

        self.set_spacing_layer()

    def set_spacing_layer(self, layer = 150, neuron = 50):
        self.gap_layer = layer
        self.gap_neuron = neuron
    
    def set_layers(self, arr_layer = [4, 6, 5]):
        self.layers_bp = arr_layer
        self.layer_num = len(arr_layer)

    def get_neuron_num(self):
        return sum(self.layers_bp)



# ----------------------------
class Neuron:
    def __init__(self, x, y, radius=10, color=(255, 0, 0)):
        self.x = x
        self.y = y

        self.radius = radius
        self.color = color
        self.column_idx = None

        self.circle = None

    def batch_vfx(self, batch):
        color=(50, 225, 30)
        circle = pglt.shapes.Circle(self.x, self.y, self.radius, color=color, batch=batch)
        self.circle = circle

    def setup_animation(self, generator, phase_delta):
        self.phase = next(generator)
        self.phase_delta = phase_delta

    def update_animation(self):
        self.x = self.x + np.sin(self.phase) * 8
        self.y = self.y + np.cos(self.phase) * 8

        self.circle.x = self.x
        self.circle.y = self.y

        self.phase += self.phase_delta

class NeuronConnetion:
    def __init__(self, neuron1: Neuron, neuron2: Neuron):
        self.neuron1 = neuron1
        self.neuron2 = neuron2

    def batch_vfx(self, batch):
        # beige=(255, 228, 181)
        color=(255, 228, 181)
        line = pglt.shapes.Line(self.neuron1.x, self.neuron1.y, self.neuron2.x, self.neuron2.y, width=2, color=color, batch=batch)
        self.line = line

    def update_connection(self):
        self.line.x = self.neuron1.x
        self.line.y = self.neuron1.y
        self.line.x2 = self.neuron2.x
        self.line.y2 = self.neuron2.y

class Layer:
    def my_neurons(self):
        return self.neurons
    
    # no self
    def _crate_neurons(self, x, y, y_off, num, sz: NeuralNetworkSize) -> list[Neuron]:
        gap = sz.gap_neuron
        neuron = []
        for i in range(num):
            _y = y + y_off + (i * gap)
            neuron.append(Neuron(x, _y))
        return neuron
    
    # no self   
    def _create_neurons(self, x, y, num, sz: NeuralNetworkSize) -> list[Neuron]:
        gap_neuron = sz.gap_neuron
        y_off = -((num - 1) * gap_neuron / 2.0)

        neurons = self._crate_neurons(x, y, y_off, num, sz)
        return neurons, len(neurons)
    
    def __init__(self, x, y, neuron_num, sz: NeuralNetworkSize):
        self.x = x
        self.y = y
        self.neurons, self.neuron_num = self._create_neurons(x, y, neuron_num, sz)
 
# ----------------------------

class LayerConnections:
    def __init__(self, layer1: Layer, layer2: Layer):
        self.layer1 = layer1
        self.layer2 = layer2
        self.neuron_connections = self.connect_layer(self.layer1, self.layer2)

    def connect_layer(self, layer1: Layer, layer2: Layer):
        connections = []
        for neuron1 in layer1.my_neurons():
            for neuron2 in layer2.my_neurons():
                connections.append(NeuronConnetion(neuron1, neuron2))

        return connections

class NNet():
    def my_neurons(self):
        return self.neurons
    
    def my_layers(self):
        return self.layers

    def __create_layers(self, x, y, x_off, sz: NeuralNetworkSize) -> list[Layer]:
        gap = sz.gap_layer
        layers: list[Layer] = []
        for i, layer_sz in enumerate(sz.layers_bp):
            _x = x + x_off + (i * gap)
            layers.append(Layer(_x, y, layer_sz, sz))
        return layers

    def _create_layers(self, x, y, sz: NeuralNetworkSize) -> list[Layer]:
        print(f"{sz}")
        gap_layer = sz.gap_layer
        layer_num = len(sz.layers_bp)
        x_off = -((layer_num - 1) * gap_layer / 2.0)

        layers: list[Layer] = self.__create_layers(x, y, x_off, sz)
        neurons = []
        for layer in layers:
            neurons.extend(layer.my_neurons())

        return layers, len(layers)
    
    def _colect_neurons(self, layers: list[Layer]):
        neurons = []
        for layer in layers:
            neurons.extend(layer.my_neurons())
        
        return neurons, len(neurons)
    
    def __init__(self, pos, nsz: NeuralNetworkSize) -> None:
        self.nsz = nsz
        x, y = pos
        self.layers, self.layer_num = self._create_layers(x, y, nsz)
        self.neurons, self.neuron_num = self._colect_neurons(self.layers)

class ConnectedLayers:
    def __init__(self, net: NNet):
        self.net = net
        self.layer_connections = self.connect_net(net)
        self.neuron_connections = self._my_neuron_connections()
    
    def connect_net(self, net: NNet) -> list[LayerConnections]:
        connections = []
        for i in range(net.layer_num - 1):
            print(f"i: {i}, {net.layer_num}")
            layer1 = net.my_layers()[i]
            layer2 = net.my_layers()[i + 1]
            connections.append(LayerConnections(layer1, layer2))
        
        return connections
    
    def _my_neuron_connections(self) -> list[NeuronConnetion]:
        neuron_connections = []
        for layer_connection in self.layer_connections:
            neuron_connections.extend(layer_connection.neuron_connections)
        
        return neuron_connections
    
    def batch_vfx(self, batch):
        for neuron_connection in self.neuron_connections:
            neuron_connection.batch_vfx(batch)
        
    
# Liczba kolumn i neuronów w każdej kolumnie
def phase_generator(n):
    cycle = 2 * np.pi
    phase = np.linspace(0, cycle, n)[0:-1]
    while True:
        for p in phase:
            yield p

def anim_realated(freq, neuron_num: int):
    anim_gen = phase_generator(neuron_num)
    phase_delta = (2 * np.pi * freq)/60

    return anim_gen, phase_delta

def spawn_nets(nsz: NeuralNetworkSize):
    spots = [(200, 300), (600, 300)]
    net1 = NNet(spots[0], nsz)

    nets = [net1]
    layers = list[Layer]()
    neurons = list[Neuron]()
    for net in nets:
        layers.extend(net.my_layers())
        neurons.extend(net.my_neurons())

    return neurons, layers, nets

def setup_neuron_rendering(neurons: list[Neuron]):
    anim_gen, phase_delta = anim_realated(2.22, len(neurons))
    neuron_draw_batch = pglt.graphics.Batch()

    for neuron in neurons:
        neuron.batch_vfx(neuron_draw_batch)
        neuron.setup_animation(anim_gen, phase_delta)

    return neuron_draw_batch

def config_size():
    nsz = NeuralNetworkSize()
    nsz.set_spacing_layer(150, 50)
    nsz.set_layers([4, 6, 5, 6, 5, 4])

    return nsz

def spawn_connections(nets: list[NNet]):
    connected_layers = ConnectedLayers(nets[0])
    layer_conn = connected_layers.layer_connections
    neuron_conn = connected_layers.neuron_connections

    return neuron_conn, layer_conn, connected_layers

def setup_connection_rendering(neuron_conn: list[NeuronConnetion]):
    connect_batch = pglt.graphics.Batch()
    for conn in neuron_conn:
        conn.batch_vfx(connect_batch)
    
    return connect_batch

def main():

    window = pglt.window.Window(800, 600)
    nsz = config_size()

    neurons, layers, nets = spawn_nets(nsz)
    neuron_batch = setup_neuron_rendering(neurons)

    neuron_conn, layer_conn, connected_layers = spawn_connections(nets)
    connection_batch = setup_connection_rendering(neuron_conn)


    update_neurons = lambda: [neuron.update_animation() for neuron in neurons]
    update_connection = lambda: [connection.update_connection() for connection in neuron_conn]

    @window.event
    def on_draw():
        window.clear()
        update_neurons()
        update_connection()
        connection_batch.draw()
        neuron_batch.draw()

    pyglet.app.run()

# if main run
if __name__ == '__main__':
    main()