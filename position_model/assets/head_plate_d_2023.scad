% scale(1000) import("head_plate_d_2023.stl");

// Sketch PureShapes 3.0
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 3.000000;
translate([0, 0, -8]) {
  translate([23.800000, 23.500000, 0]) {
    cylinder(r=23.500000,h=thickness);
  }
  translate([-40.800000, 0.000000, 0]) {
    rotate([0, 0, 0.0]) {
      cube([61.600000, 47.000000, thickness]);
    }
  }
}
}
