import coalpy.gpu as g
import numpy as nm

print("BRDF-TESTS")

g_max_zoom = 50.0
g_min_zoom = 0.05
brdf_test_shader = g.Shader(file="brdf-tests.hlsl", main_function="csBrdfTestsMain")
brdf_buff = g.Buffer(type=g.BufferType.Raw, element_count = 1)
request = None

class Params:
    def __init__(self):
        self.VdotN = 0.5
        self.roughness = 0.5
        self.scroll = (0, 0, 0)
        self.zoom = 1.0
        self.mouse_pos = (0, 0)
        self.brdf_menu = True
        self.display_menu = True
        self.brdf_integral = 0

p = Params()

def build_ui(imgui):
    global p
    imgui.begin("Params")
    if (imgui.collapsing_header("BRDF Params", g.ImGuiTreeNodeFlags.DefaultOpen)):
        p.VdotN = imgui.slider_float("VdotN", v= p.VdotN, v_min=0, v_max=1)
        p.roughness = imgui.slider_float("roughness", v= p.roughness, v_min=0, v_max=1)
    if (imgui.collapsing_header("Display", g.ImGuiTreeNodeFlags.DefaultOpen)):
        p.scroll = imgui.input_float3("scroll", v = p.scroll)
        p.zoom = imgui.slider_float("zoom", v = p.zoom, v_min=g_min_zoom, v_max=g_max_zoom)
    if (imgui.collapsing_header("Info", g.ImGuiTreeNodeFlags.DefaultOpen)):
        imgui.text("brdf integral:\t" + str(p.brdf_integral))
    imgui.end()

def parse_inputs(p, args):
    keys = args.window
    (pX, pY, nX, nY) = keys.get_mouse_position()
    t = 2.0
    if (keys.get_key_state(g.Keys.MouseRight)):
        delta = (p.mouse_pos[1] - nY)
        p.zoom += (abs(delta) ** 0.5) * (1.0 if delta > 0.0 else -1.0)
        p.zoom = max(min(p.zoom, g_max_zoom), g_min_zoom)
    elif (keys.get_key_state(g.Keys.MouseCenter)):
        p.scroll = (p.scroll[0] + t * (p.mouse_pos[0] - nX), p.scroll[1] - t * (p.mouse_pos[1] - nY), 0)
    p.mouse_pos = (nX, nY)

def on_render(args):
    global request
    global p
    parse_inputs(p, args)
    build_ui(args.imgui)
    
    cmd = g.CommandList()
    cmd.dispatch(
        shader = brdf_test_shader,
        constants = [
            float(args.width), float(args.height), float(1.0/args.width), float(1.0/args.height),
            float(p.VdotN), 0.0, 0.0, 0.0,
            float(p.roughness), 0.0, 0.0, 0.0,
            float(p.scroll[0]), float(p.scroll[1]), float(p.zoom), 0.0
        ],
        outputs = [args.window.display_texture, brdf_buff],
        x = int((args.width+7)/8),
        y = int((args.height+7)/8),
        z = 1
    )
    g.schedule(cmd)

    if (request == None):
        request = g.ResourceDownloadRequest(resource = brdf_buff)

    if (request.is_ready()):
        request.resolve()
        p.brdf_integral = nm.frombuffer(request.data_as_bytearray(), dtype='float32')[0]
        request = None


    return

w = g.Window(title="brdf-tests", width=1920, height=1080, on_render = on_render)

g.run()

