



import pyglet
from pyglet.gl import *
import pyglet as pglt
import numpy as np


from src.neuro_structure import NeuralNetworkSize, Neuron, NeuronConnetion, Layer, NNet, LayeredNetConnected
from src.neuro_boot import spawn_nets, setup_neuron_rendering
from src.neuro_boot import spawn_connections, setup_connection_rendering


def config_size():
    nsz = NeuralNetworkSize()
    nsz.set_spacing_layer(150, 50)
    nsz.set_layers([3, 4, 2])

    return nsz

def create_label(text, x, y):
    label = pyglet.text.Label(text,
                          font_name='Times New Roman',
                          font_size=36,
                          x=x, y=y,
                          anchor_x='center', anchor_y='center')
    
    return label


def main():

    window = pglt.window.Window(800, 600)
    label = create_label("Neural Network", 400, 550)
    key_pressed_label = create_label("key pressed:", 400, 500)

    nsz = config_size()

    neurons, layers, nets = spawn_nets(nsz)
    neuron_batch = setup_neuron_rendering(neurons)

    neuron_conn, layer_conn, connected_layers = spawn_connections(nets)
    connection_batch = setup_connection_rendering(neuron_conn)


    update_neurons = lambda: [neuron.update() for neuron in neurons]
    update_connection = lambda: [connection.update() for connection in neuron_conn]

    @window.event
    def on_draw():
        window.clear()
        update_neurons()
        update_connection()
        connection_batch.draw()
        neuron_batch.draw()
        label.draw()
        key_pressed_label.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        key_pressed_label.text = f"key pressed: {symbol}"

        

    pyglet.app.run()

# if main run
if __name__ == '__main__':
    main()