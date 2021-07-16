const float PI = 3.1415926535897932384626433832795;
const float TAU = 2.* PI;
uniform vec3 uColor;
uniform vec3 uPosition;
uniform vec3 uRotation;
uniform vec2 uResolution;
uniform sampler2D uTexture;
uniform vec2 uMouse;


varying float vDistort;
varying vec2 vUv;
varying float vElevation;
varying float vTime;

const int circlesArrLength = 1;
void pMod2(inout vec2 p, vec2 size){
  p = mod(p, size) -size * .005 ;
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

float wiggly(float cx, float cy, float amplitude, float frequency, float spread){

  float w = sin(cx * amplitude * frequency * PI) * cos(cy * amplitude * frequency * PI) * spread;

  return w;
}

vec3 shape( in vec2 p, in int sides )
{
  float slowTime = vTime * .05;
  float d = 0.0;
  vec2 st = p *2.-1.;

  // Number of sides of your shape
  int N = sides ;

  // Angle and radius from the current pixel
  float a = atan(st.x,st.y)+PI ;
  float r = (2.* PI)/float(N) ;

  // Shaping function that modulate the distance
  d = cos(floor(.5+a/r)*r-a)*length(st);




  return  vec3(1.0-smoothstep(.4,.81,d));
}

float triangleDF(vec2 uv){
  uv =(uv * 2. -1.) * 2.;
  return max(
    abs(uv.x) * 0.866025 + uv.y * 0.5 ,
     -1. * uv.y * 0.5);
}

float rectSDF(vec2 uv, vec2 s){
  uv = uv * 2. -1.;
  return max(
     abs(uv.x/s.x),
     abs(uv.y/s.y));
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

float stroke(float x, float s, float w){
  float d = step(s,x + w * .5) -
  step(s, x-w *.5);


  return clamp(d, 0., 1.);
}
vec3 bridge(vec3 c, float d, float s, float w){
  c*= 1. -stroke(d,s,w*2.);
  return c + stroke(d,s,w);
}

float flip(float v, float pct){
  return mix(v, 1.-v, pct);
}

float fill(float x,float size){
  return 1. -step(size,x);
}

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float spiralSDF(vec2 st, float t){
  st -= .5;
  float r = dot(st, st);
  float a = atan(st.y, st.x);
  return abs(sin(fract(log(r)*t+a*0.159)));
}

void spin(inout vec2 p, float axis){
  p.x += sin(vTime *.5) *axis;
  p.y += cos(vTime *.5) *axis;
}

void spinC(inout vec2 p, float axis){
  p.x -= sin(vTime *.5) *axis;
  p.y -= cos(vTime *.5) *axis;
}
void uvX(inout vec2 p, float axis){
  p.x -= sin(vTime *.5) *axis;

}

void uvRipple(inout vec2 uv, float intensity){

	vec2 p =-1.+2.*gl_FragCoord.xy / uResolution.xy-vec2(0,-.001);


    float cLength=length(p);

     uv= uv +(p/cLength)*cos(cLength*15.0-vTime*1.0)*intensity;

}

void uvRippleStatic(inout vec2 uv, float intensity){

	vec2 p =-1.+2.*gl_FragCoord.xy / uResolution.xy-vec2(0,-.001);


    float cLength=length(p);

     uv= uv +(p/cLength)*cos(cLength*15.0)*intensity;

}

float sphereSDF(vec3 p) {
    return length(p) - 1.0;
}

float raysSDF(vec2 uv, float N){
  uv -=.5;
  return fract(atan(uv.y, uv.x)/TAU * float(N));
}

float polySDF(vec2 uv, float sides){
  uv = uv * 2. - 1.;
  float a = atan(uv.x,uv.y) + PI;
  float r = length(uv);
  float v = TAU /sides;
  return cos(floor(.5+a/v) * v - a) * r;
}

vec2 random2(vec2 st){
    st = vec2( dot(st,vec2(127.1,311.7)),
              dot(st,vec2(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(st)*43758.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( random2(i + vec2(0.0,0.0) ), f - vec2(0.0,0.0) ),
                     dot( random2(i + vec2(1.0,0.0) ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( random2(i + vec2(0.0,1.0) ), f - vec2(0.0,1.0) ),
                     dot( random2(i + vec2(1.0,1.0) ), f - vec2(1.0,1.0) ), u.x), u.y);
}

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

#define PHI (sqrt(5)*0.5 + 0.5)

// https://math.stackexchange.com/questions/2491494/does-there-exist-a-smooth-approximation-of-x-bmod-y
// found this equation and converted it to GLSL, usually e is supposed to be squared but in this case I like the way it looks as 0 //idk
float smoothMod(float x, float y, float e){
float top = cos(PI * (x/y)) * sin(PI * (x/y));
float bot = pow(sin(PI * (x/(y+0.1))),2.)+ pow(e, sin(vTime));
float at = atan(top/bot);
return y * (1./2.) - (1./PI) * at ;
}


vec3 hsb2rgb( in vec3 c ){
vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                       6.0)-3.0)-1.0,
               0.0,
               1.0 );
rgb = rgb*rgb*(3.0-2.0*rgb);
return c.z * mix( vec3(1.0), rgb, c.y);
}

// to see this function graphed out go to: https://www.desmos.com/calculator/rz7abjujdj
vec3 cosPalette( float t )
{
vec2 normCoord = gl_FragCoord.xy/uResolution;
t = t * 0.15;
vec2 uv = -1. + 2. * normCoord;

// please play around with these numbers to get a better palette
vec3 brightness = vec3(sin(vTime), length(uv), cos(vTime));
vec3 contrast = vec3(length(uv)*.5, PI, .5);
vec3 osc = vec3(0.0,2.0,0.0);
vec3 phase = vec3(1.,12.0,6.);
return brightness + contrast*cos( 6.28318*(osc*t+phase) );
}
float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))
                 * 43758.5453123);
}

float noisePix (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

void main() {

  vec2 uv1 = vUv;
  vec2 uv2 = vUv;
  vec2 uv3 = vUv;
  vec2 uv4 = vUv;



  float slowTime = vTime * .05;

  float alpha = 1.;

vec2 uv = vUv;
vec2 rote = rotateUV(uv, vec2(.5), PI * vTime * .05);
vec2 roteC = rotateUV(uv, vec2(.5), -PI * vTime * .05);
float t = vTime ;
float scale = 5. ;
vec2 mouse = uMouse;
vec2 bulge = vec2(0.);
float d = 0.;

vec3 circle = vec3 (.5,.5, .5 +  wiggly(roteC.x + vTime * .05, roteC.y + vTime * .005, 2., 6. , .05  ) );

// uv.x += stroke(cnoise(rote * 10.), .5,.5);
// uv.y += stroke(cnoise(roteC *30.), .5, .6);
for(int i=0; i < circlesArrLength; i++)
{
float growth = 0.001 +circle.z * 8.;
float r = growth * .022;

// bulge += (uv ) * scale/float(circlesArrLength)  * pow((length(uv) ), .5);
d += smoothstep(r, r*0.5, length(uv -circle.xy )-r);
    }
uv = mix(uv * scale, bulge, d);
// uv.y -= t;
vec2 gv = fract(uv) - .5;
vec2 id = floor(uv);
// uvRipple(id, .05);
// id = brownConradyDistortion(id, -10., -1.);
float pattern = mod(id.x+id.y, 2.);
float pattern2 = mod(id.x*id.y, 3.);
float pattern3 = mod(id.x*id.y, 4.);
vec3 color = vec3(pattern);


//

vec3 pat1 = vec3(stroke(noisePix(rote * 109. * snoise(uv1) ), .2, .8), stroke(noisePix(rote * 19. * snoise(uv1) ), .2, .8), stroke(cnoise(rote * 59. * snoise(uv1) ), .2, .8) );
vec3 pat2 = vec3(1.-stroke(noisePix(rote * 19. * snoise(rote) ), .2, .8));

color = mix(pat1, pat2, pattern);
color = mix(color, vec3(cnoise(roteC * 34.)), pattern2);
color = mix(1.-color, vec3(cnoise(uv * 3. * uv.y), cnoise(uv * 7. * uv.y), cnoise(uv * 13. * uv.y)) , pattern3);

vec3 iri = hsb2rgb(vec3(color.x, color.y, uv.y *.3));
// pMod2(color.zy, vec2(.5));
iri += vec3(tan(length(iri.xz) * 4.0 + t));

color = mix(color, iri, pattern2);


 gl_FragColor = vec4(color, 1.0);



}
