
import pyglet as pglt
import numpy as np

# ===== sizing =====

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

# ===== units =====

class Neuron:
    def __init__(self, x, y, radius=10, color=(50, 225, 30)):
        self.x_root = x
        self.y_root = y

        self.x = self.x_root
        self.y = self.y_root

        self.radius = radius
        self.color = color

        self.circle = None

    def batch_vfx(self, batch):
        color=self.color
        circle = pglt.shapes.Circle(self.x_root, self.y_root, self.radius, color=color, batch=batch)
        self.circle = circle

    def setup_animation(self, generator, phase_delta):
        self.phase = next(generator)
        self.phase_delta = phase_delta

    def update(self):
        self.x = self.x_root + np.sin(self.phase) * 8
        self.y = self.y_root + np.cos(self.phase) * 8

        self.circle.x = self.x
        self.circle.y = self.y

        self.phase += self.phase_delta

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

# ===== connections =====

class NeuronConnetion:
    def __init__(self, neuron1: Neuron, neuron2: Neuron):
        self.neuron1 = neuron1
        self.neuron2 = neuron2

    def batch_vfx(self, batch):
        # beige=(255, 228, 181)
        color=(255, 228, 181)
        line = pglt.shapes.Line(self.neuron1.x_root, self.neuron1.y_root, self.neuron2.x_root, self.neuron2.y_root, width=2, color=color, batch=batch)
        self.line = line

    def update(self):
        self.line.x = self.neuron1.x
        self.line.y = self.neuron1.y
        self.line.x2 = self.neuron2.x
        self.line.y2 = self.neuron2.y

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

class LayeredNetConnected:
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
        