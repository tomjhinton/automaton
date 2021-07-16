const float PI = 3.1415926535897932384626433832795;
const float TAU = 2.* PI;
uniform vec3 uColor;
uniform vec3 uPosition;
uniform vec3 uRotation;
uniform vec2 uResolution;
uniform sampler2D uTexture;
uniform sampler2D uVideo;
uniform sampler2D uVideo2;
uniform vec2 uMouse;


varying float vDistort;
varying vec2 vUv;
varying float vElevation;
varying float vTime;

precision highp float;

#define PI 3.14159265359

float stroke(float x, float s, float w){
  float d = step(s,x + w * .5) -
  step(s, x-w *.5);


  return clamp(d, 0., 1.);
}

//	Classic Perlin 2D Noise
//	by Stefan Gustavson
//
vec4 permute(vec4 x)
{
    return mod(((x*34.0)+1.0)*x, 289.0);
}


vec2 fade(vec2 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

float cnoise(vec2 P){
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, 289.0); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);
  vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0; // 1/41 = 0.024...
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;
  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);
  vec4 norm = 1.79284291400159 - 0.85373472095314 *
    vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}


const int RAYMARCH_MAX_STEPS = 200;
const float RAYMARCH_MAX_DIST = 50.;
const float EPSILON = 0.0001;

// pos period of repition and limit
#define clamprepetition(p,per,l) p=p-per*clamp(round(p/per), -l, l)

mat2 rot (float a) {
	return mat2(cos(a),sin(a),-sin(a),cos(a));
}

float wiggly(float cx, float cy, float amplitude, float frequency, float spread){

  float w = sin(cx * amplitude * frequency * PI) * cos(cy * amplitude * frequency * PI) * spread;

  return w;
}

// p: position c: corner
float sdBox(vec3 p, vec3 c) {
  vec3 q = abs(p) - c;
  return length(max(q,cnoise(vUv * 40.))) + min(max(q.x,max(q.y,q.z)),sin(vTime * .5) );
}


vec3 pMod3(inout vec3 p, vec3 size) {
	vec3 c = floor((p + size*0.5)/size);
	p = mod(p + size*0.5, size) - size*0.5;
	return c;
}



float sdBoxFrame( vec3 p, vec3 b, float e )
{

  p = abs(p  )-b;

  vec3 q = abs(p+e)-e;
    // pMod3(q, vec3(1. , .5, 0. ));
  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

float sdPyramid( vec3 p, float h)
{
  float m2 = h*h + 0.25;

  p.xz = abs(p.xz);
  p.xz = (p.z>p.x) ? p.zx : p.xz;
  p.xz -= 0.5;

  vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);

  float s = max(-q.x,0.0);
  float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );

  float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
  float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);

  float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);

  return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
}

float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

#define PHI (pow(5.,0.5)*0.5 + 0.5)
float fBlob(vec3 p) {
  // pMod3(p, vec3(4., 4. + sin(vTime) * .5 + 1., 0.));
	p = abs(p);
	if (p.x < max(p.y, p.z)) p = p.yzx;
	if (p.x < max(p.y, p.z)) p = p.yzx;
	float b = max(max(max(
		dot(p, normalize(vec3(1, 1, 1))),
		dot(p.xz, normalize(vec2(PHI+1., 1)))),
		dot(p.yx, normalize(vec2(1, PHI)))),
		dot(p.xz, normalize(vec2(1, PHI))));
	float l = length(p);
	return l - 1.5 - 0.2 * (1.5 / 2.)* cos(min(sqrt(1.01 - b / l)*(PI / 0.25), PI))  ;
}

float pModPolar(inout vec2 p, float repetitions) {
	float angle = 2.*PI/repetitions;
	float a = atan(p.y, p.x) + angle/2.;
	float r = length(p);
	float c = floor(a/angle);
	a = mod(a,angle) - angle/2.;
	p = vec2(cos(a), sin(a))*r;
	// For an odd number of repetitions, fix cell index of the cell in -x direction
	// (cell index would be e.g. -5 and 5 in the two halves of the cell):
	if (abs(c) >= (repetitions/2.)) c = abs(c);
	return c;
}


float fOpTongue(float a, float b, float ra, float rb) {
	return min(a, max(a - ra, abs(b) - rb));
}

float fOpGroove(float a, float b, float ra, float rb) {
	return max(a, min(a + ra, rb - abs(b)));
}

float fOpUnionChamfer(float a, float b, float r) {
	return min(min(a, b), (a - r + b)*sqrt(0.5));
}



float scene(vec3 pos) {
	pos.yz *= rot(atan(1./sqrt(2.)));
	pos.xz *= rot(PI/4.);

	float period = 2.*(sin(vTime*.5)*0.5+1.);
	vec2 id = round(pos.xz/period);
	// clamprepetition(pos.yz, 4. + sin(vTime * .05), 4.); // Keep the last float as an int not a decimal float
  // pModPolar(pos.xy, 1. + sin(vTime * .5)*.5 + 1.);
  // pModPolar(pos.yz, 2. + sin(vTime * .5)*.5 + 1.);
  vec3 pos2 = pos;
  // pModPolar(pos.xz, 4. + sin(vTime * .5)*.5 + 1.);
	pos.yz *= rot(vTime*length(.5));
	// pos.x+= sin(vTime);
  // pos.yz *= rot(vTime*length(id +.2));
  // pos.z += sin(vTime * id.x);
  // pos.y += sin(vTime * id.y);
  // pos.x += cos(vTime * id.y);
  // pos.xyz += (sin(vTime) * 2.) * .1 * cos(3. * pos.yzx + vTime);
  // pos.x+= cnoise(vUv * 20. / pos.x);
  pos.xyz += 2. * .05 * cos(11. * pos.yzx + vTime);
  // pMod3(pos2, vec3(3. +cos(vTime), 3. + sin(vTime), 0. ));
  // pos2.xyz += 1. * .025 * cos(17. * pos.yzx + vTime);
  // pos.xyz += 1. * .0125 * cos(21. * pos.yzx + vTime);
	float box = sdBox(pos, vec3( 2.5 ));
  float pyramid = sdPyramid(pos, 4.1 );
  float pyramid2 = sdPyramid(pos, 3.1 );
  float torus = sdTorus(pos, vec2(.9, .5));
  float blob = fBlob(pos);


	// return  mix(box, torus, wiggly(vUv.x + vTime * .05, vUv.y + (vTime * .5) * .5, 2., .6, 1.5));
  // return mix(pyramid, fOpUnionChamfer(box, blob, sin(vTime)), .5);
  // return mix(mix(blob, box,(sin(vTime * .55))), mix(torus, pyramid,(sin(vTime * .35))), sin(vTime));
  return mix(box, blob, sin(vTime));
}



vec3 getnormalsmall (vec3 p)
{
		vec2 epsilon = vec2(0.001, 0.);
		return normalize(scene(p) - vec3(scene(p-epsilon.xyy),
										   scene(p-epsilon.yxy),
										   scene(p-epsilon.yyx))
						);
}
vec2 rotateUV(vec2 uv, vec2 pivot, float rotation) {
  mat2 rotation_matrix=mat2(  vec2(sin(rotation),-cos(rotation)),
                              vec2(cos(rotation),sin(rotation))
                              );
  uv -= pivot;
  uv= uv*rotation_matrix;
  uv += pivot;
  return uv;
}



vec4 raymarch(vec3 rayDir, vec3 pos) {
	// Define the start state
	// reset to 0 steps
	float currentDist = 0.0; // signed distance
	float rayDepth = 0.0;
	vec3 rayLength = vec3(0.0);
	vec3 light = normalize(vec3(1.,sin(vTime),4.));
  vec2 uv = vUv;
	vec3 gradient = mix(vec3(0.0, 0.0, sin(vTime)*.2), vec3(0.5, 0.0 ,0.5), rayDir.y);
  float warpsScale =  4. ;
  vec2 rote = rotateUV(uv, vec2(.5), PI * vTime * .05);
  vec2 roteC = rotateUV(uv, vec2(.5), -PI * vTime * .05);

  // vec4 bgColor = vec4(vec3(stroke(cnoise(uv * 40. * uv.x * sin(vTime * uv.y)), .5, .5),
	// stroke(cnoise(uv * 4. * uv.y * cos(vTime * uv.y)), wiggly(uv.x, uv.y, .5,.5,.5), .5),
	// stroke(cnoise(uv * 30. * uv.y * sin(vTime * uv.x)), .5, .5)), 1.);
  // vec4 bgColor = vec4(vec3(stroke(cnoise( rote * 40. * cnoise(roteC * 8. * uv.x)), .5, .1) ), 1.);


  vec4 bgColor = vec4(1.);

  // bgColor.xyz += warpsScale * .1 * cos(3. * bgColor.yzx + vTime);
  // bgColor.xyz += warpsScale * .05 * cos(11. * bgColor.yzx + vTime);
	// // vec4 bgColor = vec4(  1.);
  // bgColor.xyz += warpsScale * .025 * cos(17. * bgColor.yzx + vTime);
  // bgColor.xyz += warpsScale * .0125 * cos(21. * bgColor.yzx + vTime);

  // pos.xyz += 1. * .1 * cos(3. * pos.yzx + vTime);
  // pos.xyz += 1. * .05 * cos(11. * pos.yzx + vTime);
  // pos.xyz += 1. * .025 * cos(17. * pos.yzx + vTime);
  // pos.xyz += 1. * .0125 * cos(21. * pos.yzx + vTime);
  vec3 color1 = vec3(uv.y, uv.x, 1.);
  color1.xyz += warpsScale * .1 * cos(3. * color1.yzx + vTime);
  color1.xyz += warpsScale * .05 * cos(11. * color1.yzx + vTime);
  // color1.xyz += warpsScale * .025 * cos(17. * color1.yzx + vTime);
  // color1.xyz += warpsScale * .0125 * cos(21. * color1.yzx + vTime);
  vec3 color2 = vec3(1., uv.y, uv.x);
  color2.xyz += warpsScale * .1 * sin(3. * color2.yzx + vTime);
  color2.xyz += warpsScale * .05 * cos(11. * color2.yzx + vTime);
  // color2.xyz += warpsScale * .025 * cos(17. * color2.yzx + vTime);
  // color2.xyz += warpsScale * .0125 * cos(21. * color2.yzx + vTime);
	// shooting the ray
 	for (int i=0; i < RAYMARCH_MAX_STEPS; i++) {
     	// steps traveled
		vec3 new_p = pos + rayDir * rayDepth;
		currentDist = scene(new_p);
		rayDepth += currentDist;

		vec3 normals = getnormalsmall(new_p);
		float lighting = max(0.,dot(normals,light));



 		vec4 shapeColor = mix(
			vec4(color1, 1.),
			vec4(color2, 1.),
			lighting
		);


 	    if (currentDist < EPSILON) return shapeColor; // We're inside the scene - magic happens here
 		if (rayDepth > RAYMARCH_MAX_DIST) return bgColor; // We've gone too far
	}

	return bgColor;
}

void main() {
	vec2 uv = vUv - .5;

	vec3 camPos = vec3(uv*18. ,30.); // x, y, z axis
	vec3 rayDir = normalize(vec3(0.,0., -1.0)); // DOF

  vec4 final = vec4(raymarch(rayDir, camPos));
  // final.a = .8;
    gl_FragColor = final;
}
