



import pyglet
import numpy as np
from pyglet.gl import *
import pyglet as pglt


class NeuralNetworkSize:
    def __init__(self) -> None:
        self.gap_layer = 0
        self.gap_neuron = 0
        self.layer_s = []

        self.set_spacing_layer()

    def set_spacing_layer(self, layer = 150, neuron = 50):
        self.gap_layer = layer
        self.gap_neuron = neuron
    
    def set_layers(self, arr_layer = [4, 6, 5]):
        self.layers_bp = arr_layer
        self.numr_layers = len(arr_layer)

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
        circle = pglt.shapes.Circle(self.x, self.y, self.radius, color=(50, 225, 30), batch=batch)
        self.circle = circle

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

        return layers, len(neurons)
    
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
    
    

# Liczba kolumn i neuronów w każdej kolumnie
def spaw_newurons():
    nsz = NeuralNetworkSize()
    nsz.set_spacing_layer(150, 50)
    nsz.set_layers([4, 6, 5])


    layers = []
    neurons = []
    spots = [(200, 300), (600, 300)]
    net1 = NNet(spots[0], nsz)
    net2 = NNet(spots[0], nsz)

    nets = [net1, net2]
    layers = []
    neurons = []
    for net in nets:
        layers.extend(net.my_layers())
        neurons.extend(net.my_neurons())

    return neurons

def main():

    window = pglt.window.Window(800, 600)

    neurons = spaw_newurons()
    batch = pglt.graphics.Batch()
    for neuron in neurons:
        neuron.batch_vfx(batch)

    @window.event
    def on_draw():
        window.clear()
        batch.draw()

    pyglet.app.run()

# if main run
if __name__ == '__main__':
    main()