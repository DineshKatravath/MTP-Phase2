import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import (
    load_scene_from_string,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    RadioMaterial
)

from config import *

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS
)).astype(np.float32)


def generate_xml(mesh_path, idx):
    return f"""
    <scene version="2.1.0">

        <!-- HAND -->
        <shape type="ply">
            <string name="filename" value="{mesh_path}"/>
        </shape>

        <!-- FLOOR -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="5" z="1"/>
                <translate x="0" y="0" z="-0.15"/>
            </transform>
        </shape>

        <!-- FRONT WALL -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="1" y="0" z="0" angle="90"/>
                <translate x="0" y="0.5" z="0"/>
            </transform>
        </shape>

        <!-- LEFT WALL -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="0" y="1" z="0" angle="90"/>
                <translate x="-2.0" y="0" z="0"/>
            </transform>
        </shape>

        <!-- RIGHT WALL -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="0" y="1" z="0" angle="90"/>
                <translate x="2.0" y="0" z="0"/>
            </transform>
        </shape>

    </scene>
    """


class RFEngine:
    def __init__(self, tx, rx):
        self.tx = tx
        self.rx = rx

        # build ONCE
        self.array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )

        self.solver = PathSolver()

    def run(self, mesh_path, idx):
        scene = load_scene_from_string(generate_xml(mesh_path, idx))
        scene.frequency = FREQUENCY

        # apply material once per scene
        skin = RadioMaterial(
            name=f"skin_{idx}",
            relative_permittivity=17.3,
            conductivity=25.6,
        )

        for obj in scene.objects.values():
            obj.radio_material = skin

        scene.tx_array = self.array
        scene.rx_array = self.array

        scene.add(Transmitter(name="tx", position=self.tx))
        scene.add(Receiver(name="rx", position=self.rx))

        paths = self.solver(
            scene=scene,
            max_depth=MAX_DEPTH,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            diffraction=False,
            refraction=False,
        )

        result = paths.cfr(
            frequencies=subcarrier_freqs,
            num_time_steps=1,
            normalize_delays=True,
            normalize=False,
            out_type="numpy",
        )

        H = np.real(result).flatten()[:NUM_SUBCARRIERS] + \
            1j * np.imag(result).flatten()[:NUM_SUBCARRIERS]

        CIR = np.fft.ifft(H)

        return H, CIR