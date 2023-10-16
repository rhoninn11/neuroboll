
import pyglet as pglt
import numpy as np

from src.neuro_structure import NeuralNetworkSize
from src.neuro_structure import Neuron, Layer, NNet

# Liczba kolumn i neuronów w każdej kolumnie
def phase_generator(n, freq=1):
    cycle = 2 * np.pi * freq
    phase = np.linspace(0, cycle, n)[0:-1]
    while True:
        for p in phase:
            yield p

def anim_realated(freq, neuron_num: int):
    anim_gen = phase_generator(neuron_num, freq)
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
    anim_gen, phase_delta = anim_realated(1.33, len(neurons))
    neuron_draw_batch = pglt.graphics.Batch()

    for neuron in neurons:
        neuron.batch_vfx(neuron_draw_batch)
        neuron.setup_animation(anim_gen, phase_delta)

    return neuron_draw_batch

from src.neuro_structure import LayeredNetConnected, NeuronConnetion

def spawn_connections(nets: list[NNet]):
    connected_layers = LayeredNetConnected(nets[0])
    layer_conn = connected_layers.layer_connections
    neuron_conn = connected_layers.neuron_connections

    return neuron_conn, layer_conn, connected_layers

def setup_connection_rendering(neuron_conn: list[NeuronConnetion]):
    connect_batch = pglt.graphics.Batch()
    for conn in neuron_conn:
        conn.batch_vfx(connect_batch)
    
    return connect_batch