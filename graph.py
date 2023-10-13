import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, position, size, text):
    rect = patches.Rectangle(position, size[0], size[1], linewidth=1, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(position[0] + size[0] / 2, position[1] + size[1] / 2, text,
            horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')

def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.5))

def draw_model_structure(sensor_dims):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    # Encoder boxes for each sensor type
    num_sensors = len(sensor_dims)
    sensor_spacing = 1.2
    for i, (sensor, dim) in enumerate(sensor_dims.items()):
        draw_box(ax, (i * sensor_spacing, 4), (1, 0.8), f"Transformer\nEncoder\n{sensor} ({dim}D)")
        draw_arrow(ax, (i * sensor_spacing + 0.5, 3.2), (2.5, 2.8))

    # Concatenation & Hidden layer representation
    draw_box(ax, (1, 2.2), (3, 0.8), "Concatenate Encoded Data & Init Hidden")
    draw_arrow(ax, (2.5, 1.6), (2.5, 1.2))

    # LSTM Decoder
    draw_box(ax, (1.5, 1), (2, 0.6), "LSTM Decoder")
    draw_arrow(ax, (2.5, 0.6), (2.5, 0.4))

    # Fully Connected Layer
    draw_box(ax, (1.5, 0), (2, 0.6), "FC Layer for Text Gen")

    plt.tight_layout()
    plt.show()

sensor_dims_example = {"SensorA": 2, "SensorB": 16, "SensorC": 32, "SensorD": 66}
draw_model_structure(sensor_dims_example)
