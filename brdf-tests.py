import coalpy.gpu as g
import numpy as nm
import math

g_max_zoom = 50.0
g_min_zoom = 0.05
g_brdf_preview_shader = g.Shader(file="brdf-tests.hlsl", main_function="csBrdf2DPreview")
g_brdf_scene_shader = g.Shader(file="brdf-tests.hlsl", main_function="csRtScene")
g_brdf_integral_buff = g.Buffer(type=g.BufferType.Raw, element_count = 1)
g_request = None

class Params:
    def __init__(self):
        #brdf params
        self.VdotN = 0.5 ** 0.5
        self.roughness = 0.5

        # display params
        self.scroll = (0, 0, 0)
        self.zoom = 1.0
        self.mouse_pos = (0, 0)
        self.mouse_pos_scene_view = (0, 0)
    
        # camera
        self.reset_cam()

        # info
        self.brdf_integral = 0

    def reset_cam(self):
        self.eye_pos = (0, 0, 8.0)
        self.eye_fov_y = math.pi * 0.5 * 0.6
        self.eye_azimuth = 0
        self.eye_altitude = 0
        self.cam_speed = 0.7
        self.cam_velocity_f = 0.0
        self.cam_velocity_s = 0.0
        

p = Params()

def build_ui(imgui):
    global p
    imgui.begin("Params")
    if (imgui.collapsing_header("BRDF Params", g.ImGuiTreeNodeFlags.DefaultOpen)):
        p.VdotN = imgui.slider_float("VdotN", v= p.VdotN, v_min=0, v_max=1)
        p.roughness = imgui.slider_float("roughness", v= p.roughness, v_min=0, v_max=1)
    if (imgui.collapsing_header("3d camera", g.ImGuiTreeNodeFlags.DefaultOpen)):
        p.eye_pos = imgui.input_float3("eye pos", v = p.eye_pos)
        p.eye_fov_y = imgui.slider_float("eye fov y", v = p.eye_fov_y, v_min=0, v_max=0.5*math.pi)
        p.eye_azimuth = imgui.slider_float("eye azimuth", v = p.eye_azimuth, v_min=-math.pi, v_max=math.pi)
        p.eye_altitude = imgui.slider_float("eye altitude", v = p.eye_altitude, v_min=-0.5*math.pi, v_max=0.5 * math.pi)
        p.cam_speed = imgui.slider_float("camera speed", v = p.cam_speed, v_min=0.1, v_max=100.0)
        if (imgui.button("reset")):
            p.reset_cam()
    if (imgui.collapsing_header("Display", g.ImGuiTreeNodeFlags.DefaultOpen)):
        p.scroll = imgui.input_float3("scroll", v = p.scroll)
        p.zoom = imgui.slider_float("zoom", v = p.zoom, v_min=g_min_zoom, v_max=g_max_zoom)
    if (imgui.collapsing_header("Info", g.ImGuiTreeNodeFlags.DefaultOpen)):
        imgui.text("brdf integral:\t" + str(p.brdf_integral))
    imgui.end()

def parse_inputs_prev2d(p, args):
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

def parse_inputs_scene3d(p, args):
    keys = args.window
    (pX, pY, nX, nY) = keys.get_mouse_position()
    
    if (keys.get_key_state(g.Keys.W)):
        p.cam_velocity_f = p.cam_speed;
    elif (keys.get_key_state(g.Keys.S)):
        p.cam_velocity_f = -p.cam_speed;

    if (keys.get_key_state(g.Keys.D)):
        p.cam_velocity_s = p.cam_speed;
    elif (keys.get_key_state(g.Keys.A)):
        p.cam_velocity_s = -p.cam_speed;

    if (keys.get_key_state(g.Keys.MouseRight)):
        p.eye_altitude += (p.mouse_pos_scene_view[1] - nY);
        p.eye_azimuth += (nX - p.mouse_pos_scene_view[0]);

    if (math.fabs(p.cam_velocity_f) > 0.0 or math.fabs(p.cam_velocity_s) > 0.0):
        sa = math.sin(p.eye_altitude)
        ca = math.cos(p.eye_altitude)
        sz = math.sin(p.eye_azimuth)
        cz = math.cos(p.eye_azimuth)

        eye_z = (ca * sz, sa, -ca * cz)
        eye_x = (cz, 0, sz)

        p.eye_pos = (
            p.eye_pos[0] + eye_z[0] * p.cam_velocity_f + eye_x[0] * p.cam_velocity_s,
            p.eye_pos[1] + eye_z[1] * p.cam_velocity_f + eye_x[1] * p.cam_velocity_s,
            p.eye_pos[2] + eye_z[2] * p.cam_velocity_f + eye_x[2] * p.cam_velocity_s);

    p.cam_velocity_f = 0.0 if p.cam_velocity_f < 0.001 else 0.3 * p.cam_velocity_f;
    p.cam_velocity_s = 0.0 if p.cam_velocity_s < 0.001 else 0.3 * p.cam_velocity_s;
    p.mouse_pos_scene_view = (nX, nY)

def create_constants(p, args):
    return [
        float(args.width), float(args.height), float(1.0/args.width), float(1.0/args.height),
        float(p.VdotN), 0.0, 0.0, 0.0,
        float(p.roughness), 0.0, 0.0, 0.0,
        float(p.scroll[0]), float(p.scroll[1]), float(p.zoom), 0.0,
        float(p.eye_pos[0]), float(p.eye_pos[1]), float(p.eye_pos[2]), float(p.eye_azimuth),
        float(p.eye_altitude), float(math.cos(p.eye_fov_y)), float(math.sin(p.eye_fov_y)), 0.0
    ]

def on_render_brdf_2d_prev(args):
    global g_request
    global p
    parse_inputs_prev2d(p, args)
    cmd = g.CommandList()
    cmd.dispatch(
        shader = g_brdf_preview_shader,
        constants = create_constants(p, args),
        outputs = [args.window.display_texture, g_brdf_integral_buff],
        x = int((args.width+7)/8),
        y = int((args.height+7)/8),
        z = 1
    )
    g.schedule(cmd)

    if (g_request == None):
        g_request = g.ResourceDownloadRequest(resource = g_brdf_integral_buff)

    if (g_request.is_ready()):
        g_request.resolve()
        p.brdf_integral = nm.frombuffer(g_request.data_as_bytearray(), dtype='float32')[0]
        g_request = None
    return

def on_render_brdf_scene(args):
    global p
    parse_inputs_scene3d(p, args)
    build_ui(args.imgui)
    cmd = g.CommandList()
    cmd.dispatch(
        shader = g_brdf_scene_shader,
        constants = create_constants(p, args),
        outputs = args.window.display_texture,
        x = int((args.width+7)/8),
        y = int((args.height+7)/8),
        z = 1
    )
    g.schedule(cmd)

w0 = g.Window(title="brdf-scene", width=960, height=540, on_render = on_render_brdf_scene)
w1 = g.Window(title="brdf-2d-prev", width=400, height=400, on_render = on_render_brdf_2d_prev, use_imgui=False)

g.run()

