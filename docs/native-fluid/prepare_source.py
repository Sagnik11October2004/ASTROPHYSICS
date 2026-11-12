#!/usr/bin/env python3
"""Prepare the pinned REAL_FLUID C++ source for a headless WebAssembly build.

All dynamics, MICCG pressure projection, surface-tension terms, redistancing,
interpolation and volume controller remain from the user's original sources.
Only OpenGL visualization/printfs are disabled, and parameter API hooks and
an explicitly experimental fourth signed-distance scenario are added.

The emitted source plus this patch are copied with the compiled binary.
GPL-3.0; original: https://github.com/Sagnik11October2004/REAL_FLUID
"""
from __future__ import annotations
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve()
src = root / "src"
assert (src / "liquid2D.cpp").is_file()

# Remove graphics headers, substitute inert declarations that let the original
# source compile without EGL/GLUT. The draw functions are dead-code eliminated.
gl_names = """
glBegin glBlendFunc glColor4d glColor4f glEnable glEnd glFlush
glLoadIdentity glOrtho glPixelStorei glPointSize glPopMatrix
glPushMatrix glRasterPos2d glReadPixels glScaled glTranslated
glVertex2d glVertex2f glVertex2fv glViewport glutBitmapCharacter
""".split()
stub = [
    "#pragma once",
    "// The original renderer is replaced by an HTML Canvas view.",
    "#define GLUT_WINDOW_WIDTH 720",
    "#define GLUT_WINDOW_HEIGHT 720",
    "#define GLUT_BITMAP_HELVETICA_12 ((void*)0)",
    "inline int glutGet(int) { return 720; }",
]
for token in """
GL_BLEND GL_LINES GL_LINE_LOOP GL_ONE_MINUS_SRC_ALPHA GL_POINTS
GL_QUADS GL_RGBA GL_SRC_ALPHA GL_TRIANGLE_FAN
GL_UNPACK_ALIGNMENT GL_UNSIGNED_BYTE
""".split():
    stub.append(f"#define {token} 0")
for token in gl_names:
    stub.append(f"#define {token}(...) ((void)0)")
(src / "fluid_gl_stub.h").write_text("\n".join(stub) + "\n")

for filename in ["liquid2D.cpp", "levelset2D.cpp", "controller.cpp", "profile.cpp", "main.cpp"]:
    p = src / filename
    if not p.exists():
        continue
    s = p.read_text()
    s, n = re.subn(
        r'^[ \t]*#include[ \t]*<(?:GL|GLUT|OpenGL)/[^>]+>[ \t]*$',
        '#include "fluid_gl_stub.h"',
        s, flags=re.M,
    )
    p.write_text(s)
    print(f"GL stubbed {filename}: {n} includes")

p = src / "utility.h"
s = p.read_text()
s = s.replace("#include <omp.h>", "// No OpenMP dependency in WebAssembly build")
p.write_text(s)

p = src / "liquid2D.cpp"
s = p.read_text()
assert "static int reset_num = 3;" in s
s = s.replace("static int reset_num = 3;", "static int reset_num = 4;", 1)
# Keep the three original initial conditions unchanged; add a clearly
# experimental fourth scenario.
needle = "case 2:\n\t\t\treturn y < 0.1;"
assert needle in s
s = s.replace(
    needle, needle +
    "\n\t\tcase 3:\n\t\t\treturn y < 0.87 && hypot(x-0.5,y-0.35) > 0.115;",
    1,
)
# Expose gravity and capillary parameters. C++ original resets gravity each
# time step even at amplitude=0; replace the hard-coded base value.
needle = "static FLOAT gravity = 9.8;"
assert needle in s
s = s.replace(
    needle,
    "static FLOAT gravity = 9.8;\n\tstatic FLOAT web_base_gravity = 9.8;\n\tstatic FLOAT web_surface_tension = SURFACE_TENSION;",
    1,
)
needle = "SURFACE_TENSION / RHO"
assert needle in s
s = s.replace(needle, "web_surface_tension / RHO", 1)
needle = "gravity = 9.8 + Amplitude * sin("
assert needle in s
s = s.replace(needle, "gravity = web_base_gravity + Amplitude * sin(", 1)
# Only suppress visual drawing: the original display() still executes all
# physical steps (forces, viscosity, MICCG projection, level set evolution).
needle = "\t// Visualize Everything\n\trender();"
assert needle in s
s = s.replace(
    needle,
    "\t// WebAssembly: HTML Canvas renders the level set; physics untouched.",
    1,
)
# Avoid allocating/printing logging strings on every run step.
s = re.sub(r'^[ \t]*printf\([ \t]*"gravity = %e\\n"[ \t]*,[^\n]*\);[ \t]*$', '', s, flags=re.M)
s += '''
// WebAssembly-only parameter API.
extern "C" void fluid_set_physics(float g, float sigma) {
  if (g < 0.0f) g=0.0f;
  if (g > 30.0f) g=30.0f;
  if (sigma < 0.0f) sigma=0.0f;
  if (sigma > 0.5f) sigma=0.5f;
  web_base_gravity=g;
  gravity=g;
  web_surface_tension=sigma;
}
extern "C" float fluid_native_u(int i,int j){
  if(i<0||j<0||i>=gn||j>=gn)return 0.0f;
  return 0.5f*(u[0][i][j]+u[0][i+1][j]);
}
extern "C" float fluid_native_v(int i,int j){
  if(i<0||j<0||i>=gn||j>=gn)return 0.0f;
  return 0.5f*(u[1][i][j]+u[1][i][j+1]);
}
'''
p.write_text(s)

p = src / "solver.cpp"
s = p.read_text()
s = s.replace('#include "utility.h"', '#include "utility.h"\nstatic FLOAT web_pressure_residual = 0.0f;', 1)
needle = "return res;"
assert needle in s
s = s.replace(needle, "web_pressure_residual=res;\n\treturn res;", 1)
s += '\nextern "C" float fluid_pressure_residual(){return web_pressure_residual;}\n'
s = re.sub(r'^[ \t]*printf\([ \t]*"Residual = %e\\n"[ \t]*,[^\n]*\);[ \t]*$', '', s, flags=re.M)
p.write_text(s)
print("Prepared REAL_FLUID numerical sources with headless rendering and API hooks")
