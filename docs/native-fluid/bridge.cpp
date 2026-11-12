// Browser adapter for REAL_FLUID. Its original numerical kernels are compiled
// unmodified apart from the documented non-physics parameter hooks, headless
// graphics suppression and optional experimental fourth initial condition.
// License: GPL-3.0, following REAL_FLUID.
#include <cmath>
#include <vector>
#include "liquid2D.h"
#include "levelset2D.h"
extern "C" float fluid_native_u(int i, int j);
extern "C" float fluid_native_v(int i, int j);
extern "C" float fluid_pressure_residual();

namespace {
 int side = 64;
 int activeCase = 0;
 int pressureSolver = 2;  // original: 0 CG, 1 ICCG, 2 MICCG
 bool useRedistance = true;
 bool useAreaCorrection = true;
 int steps = 0;
 std::vector<float> image;
 std::vector<float> velocityX;
 std::vector<float> velocityY;
 void update_image() {
   if ((int)image.size() != side*side) {
     image.resize(side*side);
     velocityX.resize(side*side);
     velocityY.resize(side*side);
   }
   for(int j=0;j<side;j++)for(int i=0;i<side;i++) {
     const int k=j*side+i;
     image[k]=levelset2D::getLevelSet(i,j);
     velocityX[k]=fluid_native_u(i,j);
     velocityY[k]=fluid_native_v(i,j);
   }
 }
}
extern "C" {
// Full original C++ liquid2D::display() advances one numerical time step;
// its original rasterizer is headlessly suppressed and replaced by HTML Canvas.
void fluid_init(int n) {
  if(n < 16 || n > 256) n=64;
  side=n;
  liquid2D::init(side);
  steps=0;
  update_image();
}
int fluid_grid(){return side;}
float fluid_dt(){return 0.01f;}
float fluid_time(){return steps*0.01f;}
int fluid_case_current(){return activeCase;}
int fluid_phi_ptr(){return (int)(long)image.data();}
int fluid_u_ptr(){return (int)(long)velocityX.data();}
int fluid_v_ptr(){return (int)(long)velocityY.data();}
float fluid_volume(){return levelset2D::getVolume();}
int fluid_run_steps(int n) {
  if(n<0)n=0;
  if(n>4)n=4;
  for(int k=0;k<n;k++) {
    liquid2D::display();
    steps++;
  }
  if(n>0)update_image();
  return steps;
}
void fluid_reset() {
  liquid2D::init(side);
  steps=0;update_image();
}
// The C++ "r" key cycles its original three cases plus the experimental
// fourth signed-distance cavity added only by the web build patch.
void fluid_choose_case(int target) {
  if(target<0 || target>3) return;
  while(activeCase!=target) {
    liquid2D::keyDown('r');
    activeCase=(activeCase+1)%4;
  }
  liquid2D::init(side);
  steps=0;update_image();
}
void fluid_toggle_volume(int enabled) {
  const bool desired=enabled!=0;
  if(desired!=useAreaCorrection) {
    liquid2D::keyDown('c');
    useAreaCorrection=desired;
  }
}
void fluid_toggle_redistance(int enabled) {
  const bool desired=enabled!=0;
  if(desired!=useRedistance){
    liquid2D::keyDown('a');
    useRedistance=desired;
  }
}
void fluid_set_solver(int target){
  if(target<0 || target>2)return;
  while(pressureSolver!=target){
    liquid2D::keyDown('s');
    pressureSolver=(pressureSolver+1)%3;
  }
}
void fluid_set_interpolation(int spline){
  // Source defaults to Catmull-Rom (1), can toggle linear (0).
  static int interp=1;
  const int desired=spline?1:0;
  if(interp!=desired){liquid2D::keyDown('i');interp=desired;}
}
void fluid_set_physics(float gravity, float surface_tension);
}
