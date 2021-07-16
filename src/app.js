import './style.scss'
import * as THREE from 'three'

import { gsap } from 'gsap'

import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import Prism from 'prismjs'

import * as Tone from 'tone'

import vertexShader from './shaders/vertex.glsl'

import fragmentShader1 from './shaders/fragment-1.glsl'

import fragmentShader2 from './shaders/fragment-2.glsl'

import fragmentShader3 from './shaders/fragment-3.glsl'

import fragmentShader4 from './shaders/fragment-4.glsl'

import fragmentShader5 from './shaders/fragment-5.glsl'

import fragmentShader6 from './shaders/fragment-6.glsl'

import fragmentShader7 from './shaders/fragment-7.glsl'

import fragmentShader8 from './shaders/fragment-8.glsl'

import fragmentShader9 from './shaders/fragment-9.glsl'

import fragmentShader10 from './shaders/fragment-10.glsl'

import fragmentShader11 from './shaders/fragment-11.glsl'

import fragmentShader12 from './shaders/fragment-12.glsl'

import fragmentShader13 from './shaders/fragment-13.glsl'

import fragmentShader14 from './shaders/fragment-14.glsl'

const snippet = document.getElementById('snipp')



const fragArray = [fragmentShader1, fragmentShader2, fragmentShader3, fragmentShader4, fragmentShader5, fragmentShader6, fragmentShader7, fragmentShader8, fragmentShader9, fragmentShader10, fragmentShader11, fragmentShader12, fragmentShader13, fragmentShader14]

// let selected = Math.floor(Math.random() * fragArray.length )

let selected = fragArray.length -1

snippet.textContent = fragArray[selected]
const points =[
  {
    position: new THREE.Vector3(3.55, 0.3, -6.6),
    element: document.querySelector('.point-0')
  },
  {
    position: new THREE.Vector3(3.55, -2.3, -6.6),
    element: document.querySelector('.point-1')
  }

]

console.log(Prism)
Prism.highlightAll()
document.onkeydown = checkKey



function resetL(){
  gsap.to(left.position, { duration: .5, y: left.position.y + 0.005, delay: 0, onComplete: buttonStill  })
}

function resetR(){
  gsap.to(right.position, { duration: .5, y: right.position.y + 0.005, delay: 0, onComplete: buttonStill })
}


function buttonStill(){
  buttonMoving = false
}

const synth =  new Tone.FMSynth().toDestination();

let buttonMoving = false

function scrollLeft(){
  // synth.triggerAttackRelease("C4", "8n");
  snippet.textContent = fragArray[selected]
  Prism.highlightAll()
  if(!buttonMoving){
    buttonMoving = true
    gsap.to(left.position, { duration: .5, y: left.position.y - 0.005, delay: 0, onComplete: resetL })
  }
  // left.position.y -=.001
  if(selected > 0){
    selected --
    shaderMaterial.needsUpdate=true

    shaderMaterial.fragmentShader = fragArray[selected]
  } else if(selected === 0){
    selected = fragArray.length -1
    shaderMaterial.needsUpdate=true

    shaderMaterial.fragmentShader = fragArray[selected]
  }
}

function scrollRight(){
  // synth.triggerAttackRelease("E4", "8n");
  snippet.textContent = fragArray[selected]
  Prism.highlightAll()
  if(!buttonMoving){
    buttonMoving = true
    gsap.to(right.position, { duration: .5, y: right.position.y - 0.005, delay: 0, onComplete: resetR })
  }
  if(selected < fragArray.length -1){
    selected ++
    shaderMaterial.needsUpdate=true

    shaderMaterial.fragmentShader = fragArray[selected]
  } else  if(selected === fragArray.length -1){
    selected = 0
    shaderMaterial.needsUpdate=true

    shaderMaterial.fragmentShader = fragArray[selected]
  }
}


function checkKey(e) {
  e.preventDefault()
  e = e || window.event
  console.log(e)
  if (e.keyCode === 38) {
    // up arrow
    // console.log(selected)
  } else if (e.keyCode === 40) {
    // down arrow
    // console.log(fragArray[selected])
  } else if (e.keyCode === 37) {
    // left arrow
    scrollLeft()

  } else if (e.keyCode === 39) {
    // right arrow
    // console.log(selected)

    scrollRight()

  }

}

var modal = document.getElementById('myModal')

var refresh = document.getElementById('refresh')

refresh.onclick = function(){
  scrollRight()
}

// Get the button that opens the modal
var btn = document.getElementById('myBtn')

// Get the <span> element that closes the modal
var span = document.getElementsByClassName('close')[0]

// When the user clicks on the button, open the modal
btn.onclick = function() {
  modal.style.display = 'block'
}

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
  modal.style.display = 'none'
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function(event) {
  if (event.target === modal) {
    modal.style.display = 'none'
  }
}

const canvas = document.querySelector('canvas.webgl')

const scene = new THREE.Scene()
// scene.background = new THREE.Color( 0xffffff )
const loadingBarElement = document.querySelector('.loading-bar')
const loadingBarText = document.querySelector('.loading-bar-text')
const loadingManager = new THREE.LoadingManager(
  // Loaded
  () =>{
    window.setTimeout(() =>{
      gsap.to(overlayMaterial.uniforms.uAlpha, { duration: 3, value: 0, delay: 1 })

      loadingBarElement.classList.add('ended')
      loadingBarElement.style.transform = ''

      loadingBarText.classList.add('fade-out')

    }, 500)
  },

  // Progress
  (itemUrl, itemsLoaded, itemsTotal) =>{
    const progressRatio = itemsLoaded / itemsTotal
    loadingBarElement.style.transform = `scaleX(${progressRatio})`

  }
)

const gtlfLoader = new GLTFLoader(loadingManager)

const overlayGeometry = new THREE.PlaneGeometry(2, 2, 1, 1)
const overlayMaterial = new THREE.ShaderMaterial({
  depthWrite: false,
  uniforms:
    {
      uAlpha: { value: 1 }
    },
  transparent: true,
  vertexShader: `
        void main()
        {
            gl_Position = vec4(position, 1.0);
        }
    `,
  fragmentShader: `
  uniform float uAlpha;
        void main()
        {
            gl_FragColor = vec4(0.0, 0.0, 0.0, uAlpha);
        }
    `
})

const overlay = new THREE.Mesh(overlayGeometry, overlayMaterial)
scene.add(overlay)


const shaderMaterial  = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: true,
  uniforms: {
    uTime: { value: 0},
    uResolution: { type: 'v2', value: new THREE.Vector2() }
  },
  vertexShader: vertexShader,
  fragmentShader: fragArray[selected],
  side: THREE.DoubleSide
})
// console.log(shaderMaterial)
let sceneGroup, left, right, displayScreen, display

const intersectsArr = []
gtlfLoader.load(
  'display.glb',
  (gltf) => {
    // console.log(gltf)
    gltf.scene.scale.set(4.5,4.5,4.5)
    sceneGroup = gltf.scene
    sceneGroup.needsUpdate = true
    sceneGroup.position.y -= 3
    scene.add(sceneGroup)



    left = gltf.scene.children.find((child) => {
      return child.name === 'Left'
    })

    right = gltf.scene.children.find((child) => {
      return child.name === 'Right'
    })

    displayScreen = gltf.scene.children.find((child) => {
      return child.name === 'Screen'
    })

    display = gltf.scene.children.find((child) => {
      return child.name === 'Body'
    })
    intersectsArr.push(left.children[0], left.children[1], right.children[0], right.children[1], displayScreen)
    displayScreen.needsUpdate = true

    displayScreen.material = shaderMaterial


  }
)


const light = new THREE.AmbientLight( 0x404040 ) // soft white light
scene.add( light )

const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.5 )
scene.add( directionalLight )

const sizes = {
  width: window.innerWidth,
  height: window.innerHeight
}

window.addEventListener('resize', () =>{



  // Update sizes
  sizes.width = window.innerWidth
  sizes.height = window.innerHeight

  // Update camera
  camera.aspect = sizes.width / sizes.height
  camera.updateProjectionMatrix()

  // Update renderer
  renderer.setSize(sizes.width, sizes.height)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2 ))


})


/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(45, sizes.width / sizes.height, 0.1, 100)
camera.position.x = 10
camera.position.y = -10
camera.position.z = 15
scene.add(camera)

// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
controls.maxPolarAngle = Math.PI / 2 - 0.1
//controls.enableZoom = false;

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
  canvas: canvas,
  antialias: true,
  alpha: true
})
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setClearColor( 0x000000, 0 )
const raycaster = new THREE.Raycaster()
const mouse = new THREE.Vector2()

renderer.domElement.addEventListener( 'pointerdown', onClick, false )

function onClick() {
  event.preventDefault()
  // console.log(mouse.x)
  mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1
  mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1
  // console.log(mouse.x)
  raycaster.setFromCamera( mouse, camera )

  var intersects = raycaster.intersectObjects( intersectsArr, true )

  if ( intersects.length > 0 ) {
    // console.log( 'Intersection:', intersects[0].object.parent.name );

    if(intersects[0].object.parent.name === 'Left'){
      scrollRight()
    }
    if(intersects[0].object.parent.name === 'Right'){
      scrollRight()
    }
  }


}


const clock = new THREE.Clock()

const tick = () =>{
  // if ( mixer ) mixer.update( clock.getDelta() )
  const elapsedTime = clock.getElapsedTime()



  if(sceneGroup){
    // sceneGroup.rotation.y += .001
    displayScreen.needsUpdate = true
  }
  if(sceneGroup){
    for(const point of points){
      const screenPosition = point.position.clone()
      screenPosition.project(camera)
      raycaster.setFromCamera(screenPosition, camera)

      const intersects = raycaster.intersectObjects(scene.children, true)
      if(intersects.length === 0){
        point.element.classList.add('visible')
      }else{
        const intersectionDistance  = intersects[0].distance
        const pointDistance = point.position.distanceTo(camera.position)
        if(intersectionDistance < pointDistance){
          point.element.classList.remove('visible')
        } else {
          point.element.classList.add('visible')
        }

      }

      const translateX = screenPosition.x * sizes.width * 0.5
      const translateY = - screenPosition.y * sizes.height * 0.5
      point.element.style.transform = `translate(${translateX}px, ${translateY}px)`

    }

  }


  // Update controls
  controls.update()

  shaderMaterial.uniforms.uTime.value = elapsedTime
  shaderMaterial.fragmentShader = fragArray[selected]




  // Render
  renderer.render(scene, camera)

  // Call tick again on the next frame
  window.requestAnimationFrame(tick)
}

tick()
