import './style.scss'

import * as THREE from 'three'

import { gsap } from 'gsap'

import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'


import vertexShader from './shaders/vertex.glsl'

import fragmentShader1 from './shaders/fragment-1.glsl'

import fragmentShader2 from './shaders/fragment-2.glsl'

import fragmentShader3 from './shaders/fragment-3.glsl'

import fragmentShader4 from './shaders/fragment-4.glsl'

import fragmentShader5 from './shaders/fragment-5.glsl'

import fragmentShader6 from './shaders/fragment-6.glsl'

import fragmentShader7 from './shaders/fragment-7.glsl'




let fragArray = [fragmentShader1, fragmentShader2, fragmentShader3, fragmentShader4, fragmentShader5, fragmentShader6, fragmentShader7]

let selected = Math.floor(Math.random() * fragArray.length )


document.onkeydown = checkKey;

function resetL(){
  gsap.to(left.position, { duration: .5, y: left.position.y + 0.005, delay: 0 })
}

function resetR(){
  gsap.to(right.position, { duration: .5, y: right.position.y + 0.005, delay: 0 })
}

function scrollLeft(){
  gsap.to(left.position, { duration: .5, y: left.position.y - 0.005, delay: 0, onComplete: resetL })
  // left.position.y -=.001
  if(selected > 0){
       selected --
     shaderMaterial.needsUpdate=true

     shaderMaterial.fragmentShader = fragArray[selected]
  }

  else if(selected === 0){
    selected = fragArray.length -1
     shaderMaterial.needsUpdate=true

     shaderMaterial.fragmentShader = fragArray[selected]
  }
}

function scrollRight(){
    gsap.to(right.position, { duration: .5, y: right.position.y - 0.005, delay: 0, onComplete: resetR })
  if(selected < fragArray.length -1){
      selected ++
      shaderMaterial.needsUpdate=true

     shaderMaterial.fragmentShader = fragArray[selected]
  }

else  if(selected === fragArray.length -1){
  selected = 0
    shaderMaterial.needsUpdate=true

     shaderMaterial.fragmentShader = fragArray[selected]
  }
}


function checkKey(e) {
e.preventDefault()
    e = e || window.event;

    if (e.keyCode == '38') {
        // up arrow
        // console.log(selected)
    }
    else if (e.keyCode == '40') {
        // down arrow
        // console.log(fragArray[selected])
    }
    else if (e.keyCode == '37') {
       // left arrow
       scrollLeft()





    }
    else if (e.keyCode == '39') {
       // right arrow
       // console.log(selected)

        scrollRight()

    }

}


const canvas = document.querySelector('canvas.webgl')

const scene = new THREE.Scene()
scene.background = new THREE.Color( 0xffffff )
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

let intersectsArr = []
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
intersectsArr.push(left.children[0], left.children[1], right.children[0], right.children[1])
 displayScreen.needsUpdate = true
 // console.log(left)




    // display.material = new THREE.MeshPhongMaterial( {color: 'black'})
    //
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
  antialias: true
})
renderer.outputEncoding = THREE.sRGBEncoding
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2()

renderer.domElement.addEventListener( 'click', onClick, false );

function onClick() {
	event.preventDefault();
// console.log(intersectsArr)
	mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

	raycaster.setFromCamera( mouse, camera );

	var intersects = raycaster.intersectObjects( intersectsArr, true );

	if ( intersects.length > 0 ) {
	    // console.log( 'Intersection:', intersects[0].object.parent.name );

      if(intersects[0].object.parent.name === 'Left'){
        scrollLeft()
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
