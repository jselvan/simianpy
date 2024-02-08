import json
import warnings
import pandas as pd

class Layer:
    def __init__(self, name, channels):
        self.name = name
        self.channels = channels
    def connect(self, interface, output):
        new_mapping_data = []
        for channel in self.channels:
            for interface_channel in interface.channels:
                if channel['loc'] == interface_channel['loc']:
                    for output_channel in output.channels:
                        if interface_channel['name'] == output_channel['name']:
                            new_mapping_data.append({"name": channel['name'], "loc": output_channel['loc']})
                            break
                    else:
                        warnings.warn(f"Channel {interface_channel['name']} not found in output {output.name}")
                    break
            else:
                warnings.warn(f"Channel {channel['name']} not found in interface {interface.name}")
        return Layer(self.name + interface.name + output.name, new_mapping_data)
    @classmethod
    def from_json(cls, jsonpath):
        with open(jsonpath, 'r') as f:
            data = json.load(f)
        return cls(data['name'], data['channels'])
    @classmethod
    def from_raw(cls, name, rawpath):
        with open(rawpath, 'r') as f:
            lines = [
                line
                for line in f.read().splitlines()
                if len(line) and not line.startswith('#')
            ]
            data = [
                {'name': int(channel.strip()), 'loc': [row, col]} 
                for row, line in enumerate(lines)
                for col, channel in enumerate(line.split(','))
                if channel.strip()
            ]
        return cls(name, data)
    def to_json(self, jsonpath):
        with open(jsonpath, 'w') as f:
            json.dump({'name': self.name, 'channels': self.channels}, f)
    def draw(self, **kwargs):
        import numpy as np
        import cv2

        # Define a font
        font = getattr(cv2, kwargs.get('font', 'FONT_HERSHEY_SIMPLEX'))
        font_scale = kwargs.get('font_scale', 0.5)
        font_thickness = kwargs.get('font_thickness', 1)
        font_color = kwargs.get('font_color', (0, 0, 0))

        width, height = kwargs.get('size', (500, 500))
        rows = max([channel['loc'][0] for channel in self.channels]) + 1
        cols = max([channel['loc'][1] for channel in self.channels]) + 1
        cell_width = width // cols
        cell_height = height // rows

        image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
        for channel in self.channels:
            row, col = channel['loc']
            text = str(channel['name'])
            x_center = col * cell_width + cell_width // 2
            y_center = row * cell_height + cell_height // 2

            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            x_text = x_center - text_size[0] // 2
            y_text = y_center + text_size[1] // 2

            # Render the text on the image
            cv2.putText(image, text, (x_text, y_text), font, font_scale, font_color, font_thickness)
        
        return image
    def show(self, **kwargs):
        from PIL import Image
        Image.fromarray(self.draw(**kwargs)).show()
    def to_csv(self, csvpath):
        data = []
        for channel in self.channels:
            x, y = channel['loc']
            data.append({'name': channel['name'], 'x': x, 'y': y})
        pd.DataFrame(data).to_csv(csvpath, index=False)



# # brm = Layer.from_json(rmapping_path/"heckles_channels.json")
# # brc = Layer.from_json(rmapping_path/"blackrock_connector.json")
# # oe = Layer.from_json(rmapping_path/"oe_connector_CBA.json")
# # # new = oe.chain_by_loc(brc).chain_by_name(brm)
# # new = oe.connect(brc, brm)
# # new.show()

# from pathlib import Path

# mapping_path = Path("D:/OneDrive/Research/Code/simianpy/test/mappings")

# dina_brm = Layer.from_raw("dina_channels", mapping_path/"dina.txt")
# brc = Layer.from_json(mapping_path/"blackrock_connector.json")
# sg = Layer.from_raw("spikegadgets", mapping_path/"spikegadgets_connector.txt")
# dina_mapping = sg.connect(brc, dina_brm)
# # dina_mapping.show()
# dina_mapping.to_csv(mapping_path/"dina_mapping.csv")

# heckles_brm = Layer.from_raw("heckles_channels", mapping_path/"heckles.txt")
# heckles_mapping = sg.connect(brc, heckles_brm)
# heckles_mapping.show()
# heckles_mapping.to_csv(mapping_path/"heckles_mapping.csv")




# # heckles_brm = Layer.from_json(mapping_path/"heckles_channels.json")

# sg_rot = Layer.from_raw("spikegadgets", mapping_path/"spikegadgets_connector_rotated.txt")
# heckles_mapping = sg_rot.connect(brc, heckles_brm)
# heckles_mapping.show()
# heckles_mapping.to_csv(mapping_path/"heckles_mapping.csv")



# dina_mapping = sg_rot.connect(brc, dina_brm)
# dina_mapping.show()
# dina_mapping.to_csv(mapping_path/"dina_mapping.csv")
