from matplotlib import pyplot as plt, patches


class DrawImageWithBoxes:
    def __init__(self, colors=None):
        if colors is None:
            colors = ["r", "g", "b"]
        self.colors = colors

    def __call__(self, image, boxes, labels):
        # Setup figure
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Plot image
        ax.imshow(image, cmap='gray')

        # Draw boxes
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=self.colors[label],
                                     facecolor='none')
            ax.add_patch(rect)

        return fig
