const float PI = 3.1415926535897932384626433832795;
const float TAU = 2.* PI;
uniform vec3 uColor;
uniform vec3 uPosition;
uniform vec3 uRotation;
uniform vec2 uResolution;
// uniform sampler2D uTexture;
// uniform sampler2D uVideo;
// uniform sampler2D uVideo2;
uniform vec2 uMouse;


varying float vDistort;
varying vec2 vUv;
varying float vElevation;
varying float vTime;
varying vec3 vNorm;

precision highp float;

#define PI 3.14159265359

vec2 brownConradyDistortion(in vec2 uv, in float k1, in float k2)
{
    uv = uv * 2.0 - 1.0;	// brown conrady takes [-1:1]

    // positive values of K1 give barrel distortion, negative give pincushion
    float r2 = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.0 + k1 * r2 + k2 * r2 * r2;

    // tangential distortion (due to off center lens elements)
    // is not modeled in this function, but if it was, the terms would go here

    uv = (uv * .5 + .5);	// restore -> [0:1]
    return uv;
}

float smoothIntersectSDF(float distA, float distB, float k )
{
  float h = clamp(0.5 - 0.5*(distA-distB)/k, 0., 1.);
  return mix(distA, distB, h ) + k*h*(1.-h);
}

// float smoothUnionSDF(float distA, float distB, float k ) {
//   float h = clamp(0.5 + 0.5*(distA-distB)/k, 0., 1.);
//   return mix(distA, distB, h) - k*h*(1.-h);
// }

// vec4 smoothDifferenceSDF(vec4 a, vec4 b, float k)
// {
//   float h = clamp(0.5 - 0.5*(a.w+b.w)/k, 0., 1.);
//   vec3 c = mix(a.rgb,b.rgb,h);
//   float d = mix(a.w, -b.w, h ) + k*h*(1.-h);
//
//   return vec4(c,d);
// }

float smoothDifferenceSDF(float distA, float distB, float k) {
  float h = clamp(0.5 - 0.5*(distB+distA)/k, 0., 1.);
  return mix(distA, -distB, h ) + k*h*(1.-h);
}

vec4 smoothUnionSDF(vec4 a, vec4 b, float k )
{
  float h = clamp(0.5 + 0.5*(a.w-b.w)/k, 0., 1.);
  vec3 c = mix(a.rgb,b.rgb,h);
  float d = mix(a.w, b.w, h) - k*h*(1.-h);

  return vec4(c,d);
}

vec4 sdSphere(vec3 p, float r, vec3 offset, vec3 col )
{
  float d = length(p - offset) - r;
  return vec4(d, col);
}

vec4 sdBox(vec3 p, vec3 c, vec3 col) {
  vec3 q = abs(p) - c;
  return vec4(length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) , col);
}



vec4 sdFloor(vec3 p, vec3 col) {
  float d = p.y + 1.;
  return vec4(d, col);
}

vec4 minWithColor(vec4 obj1, vec4 obj2) {
  if (obj2.x < obj1.x) return obj2; // The x component of the object holds the "signed distance" value
  return obj1;
}

// vec4 smoothDifferenceSDF(vec4 a, vec4 b, float k)
// {
//   float h = clamp(0.5 - 0.5*(a.w+b.w)/k, 0., 1.);
//   vec3 c = mix(a.rgb,b.rgb,h);
//   float d = mix(a.w, -b.w, h ) + k*h*(1.-h);
//
//   return vec4(c,d);
// }

mat2 rot (float a) {
	return mat2(cos(a),sin(a),-sin(a),cos(a));
}


vec4 intersectSDF(vec4 a, vec4 b) {
    return a.w > b.w ? a : b;
}

vec4 unionSDF(vec4 a, vec4 b) {
    return a.w < b.w? a : b;
}

vec4 differenceSDF(vec4 a, vec4 b) {
    return a.w > -b.w? a : vec4(b.rgb,-b.w);
}

// float opRep( in vec3 p, in vec3 c, in sdf3d primitive )
// {
//     vec3 q = mod(p+0.5*c,c)-0.5*c;
//     return primitive( q );
// }
vec3 opRep( vec3 p, vec3 c ) {
  float displacement = sin(9. * p.x + vTime * .5) * sin(3. * p.y + vTime) * sin(3. * p.z + vTime) * 0.25 ;
    vec3 q = mod(p,c)-0.5*c;
    return q * 1.;
}

vec4 sdScene(vec3 p) {
  // vec3 p2 = p;
    vec3 p3 = p;

  // p.xyz += 1. * .1 * cos(3. * p.yzx + vTime);
  // p.xyz += 1. * .05 * cos(11. * p.yzx + vTime);
  // p.xyz += 1. * .025 * cos(17. * p.yzx + vTime);
  // //
  // p3.xyz += 1.5 * .1 * cos(3. * p3.yzx + vTime);
  // p3.xyz += 1.5 * .05 * sin(11. * p3.yzx + vTime);
  // p3.xyz += 1.5 * .025 * cos(17. * p3.yzx + vTime);

  float warpsScale = 3.;
  vec3 color1 = vec3(1., vUv.y, vUv.x);

  color1.xyz += warpsScale * .1 * cos(3. * color1.yzx + vTime);
  color1.xyz += warpsScale * .05 * cos(11. * color1.yzx + vTime);
  color1.xyz += warpsScale * .025 * cos(17. * color1.yzx + vTime);
  color1.xyz += warpsScale * .0125 * cos(21. * color1.yzx + vTime);



  vec3 color2 = vec3(1., 1., 1.);
  // color2.xyz += warpsScale * .1 * sin(3. * color2.yzx + vTime);
  // color2.xyz += warpsScale * .05 * cos(11. * color2.yzx + vTime);
  // color2.xyz += warpsScale * .025 * cos(17. * color2.yzx + vTime);
  // color2.xyz += warpsScale * .0125 * cos(21. * color2.yzx + vTime);

  vec3 color3 = vec3(.5, vUv.x, 1.);
  color3.xyz += warpsScale * .1 * sin(3. * color3.yzx + vTime);
  color3.xyz += warpsScale * .05 * cos(11. * color3.yzx + vTime);
  color3.xyz += warpsScale * .025 * cos(17. * color3.yzx + vTime);
  color3.xyz += warpsScale * .0125 * cos(21. * color3.yzx + vTime);

  vec3 color4 = vec3(vUv.y, .4, 1.);
  color4.xyz += warpsScale * .1 * sin(3. * color4.yzx + vTime);
  color4.xyz += warpsScale * .05 * cos(11. * color4.yzx + vTime);
  color4.xyz += warpsScale * .025 * cos(17. * color4.yzx + vTime);
  color4.xyz += warpsScale * .0125 * cos(21. * color4.yzx + vTime);

  float displacement = sin(3. * p.x + vTime * .5) * sin(3. * p.y + vTime) * sin(3. * p.z + vTime) * 0.25 ;
  float displacement2 = cos(6.0 * p3.x + vTime) * sin(6.0 * p3.y + vTime) * sin(6.0 * p3.z + vTime) * 0.25 ;

  float displacement3 = sin(9. * p.x + vTime * .5) * sin(9. * p.y + vTime) * sin(9. * p.z + vTime) * 0.25 ;

  float displacement4 = cos(18.0 * p3.x + vTime) * sin(3.0 * p3.y + vTime) * sin(0.5 * p3.z + vTime)  ;

  vec2 thing = brownConradyDistortion(vUv, sin(vTime * .5) * (20. + displacement4), cos(vTime * .5) * (20. + displacement));



  // p3 = opRep(p3, vec3( 1. + thing.x ,1. +thing.y, 0.));



 // p3.xy *= rot(vTime * .5);
 // p3.xz *= rot(vTime * .5);
 // p.xy *= rot(vTime * .5 * -1.);

  vec4 sphereLeft = sdSphere(p3 + displacement2 ,1.1 , vec3(0., 0, .0), color1 );
  vec4 sphereRight = sdSphere(p3 + displacement4, 1., vec3(0., 0, 0.), color3);
  vec4 sphereTop = sdSphere(p3 + displacement, .5, vec3(0., -1., -2), color3);
  vec4 sphereBot = sdSphere(p3 + displacement, .5, vec3(0., 1., -2), color1);
  vec4 cube = sdBox(p3    , vec3(cos(vTime) * .5 + 1.2, .5, sin(vTime * .5) * .5 + 1.2) , color2 + displacement);
  vec4 cube2 = sdBox(p3 + displacement   , vec3(.3, sin(vTime) *.5 + .5, 1. + sin(vTime) *.5 + .5) , color4 );
  // vec4 co = mix(mix(sphereLeft, cube, sin(vTime* .8)), mix(sphereRight, cube, cos(vTime)), tan(vTime * .5));;
  // co = sphereRight;
   // co = minWithColor(co, sphereTop);
   // co = minWithColor(co, sphereBot);
   // co = closest object containing "signed distance" and color
  // co = minWithColor(co, sdFloor(p, vec3(1, .5, 0)));


  float blah = mix(cube.r, sphereLeft.r, tan(vTime * .1));
   // blah = smoothDifferenceSDF(blah, sphereLeft.r, .05);
  return vec4(blah, color2);
}

const int MAX_MARCHING_STEPS = 50;
const float MIN_DIST = 0.0;
const float MAX_DIST = 150.0;
const float PRECISION = 0.001;


vec4 rayMarch(vec3 ro, vec3 rd, float start, float end) {
  float depth = start;
  vec4 co; // closest object

  for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    vec3 p = ro + depth * rd;
    co = sdScene(p);
    depth += co.x;
    if (co.x < PRECISION || depth > end) break;
  }

  vec3 col = vec3(co.yzw);

  return vec4(depth, col);
}

vec3 calcNormal(in vec3 p) {
    vec2 e = vec2(1.0, -1.0) * 0.0005; // epsilon
    return normalize(
      e.xyy * sdScene(p + e.xyy).x +
      e.yyx * sdScene(p + e.yyx).x +
      e.yxy * sdScene(p + e.yxy).x +
      e.xxx * sdScene(p + e.xxx).x);
}

void main( )
{
  vec2 uv = vUv -.5;

  vec3 backgroundColor = vec3(0.);
  // backgroundColor.rb = brownConradyDistortion(vUv, sin(vTime * .5) * 2., cos(vTime * .5) * 2.);
  //
  // backgroundColor.rg = brownConradyDistortion(backgroundColor.rb, sin(vTime * .9) * 2., cos(vTime * .5) * 4.);

  // backgroundColor.xyz += 4. * .1 * cos(3. * backgroundColor.yzx + vTime);
  // backgroundColor.xyz += 4. * .05 * cos(11. * backgroundColor.yzx + vTime);
  vec3 lightPosition = vec3(2, 2, 7);
  vec3 col = vec3(0);
  vec3 ro = vec3(0, 0, 5); // ray origin that represents camera position
  vec3 rd = normalize(vec3(uv, -1)); // ray direction

  vec4 co = rayMarch(ro, rd, MIN_DIST, MAX_DIST); // closest object

  if (co.x > MAX_DIST) {
    col = backgroundColor; // ray didn't hit anything
  } else {
    vec3 p = ro + rd * co.x; // point on sphere or floor we discovered from ray marching
    vec3 normal = calcNormal(p);

    vec3 lightDirection = normalize(lightPosition - p);

    // Calculate diffuse reflection by taking the dot product of
    // the normal and the light direction.
    float dif = clamp(dot(normal, lightDirection), 0.3, 1.);

    // Multiply the diffuse reflection value by an orange color and add a bit
    // of the background color to the sphere to blend it more with the background.
    col = dif * co.yzw + backgroundColor * .2;
    // col = dif * co.yzw ;
  }

  // Output to screen
  gl_FragColor = vec4(col, 1.0);
}
