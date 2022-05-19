import React, { useState, useEffect, useRef} from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Dimensions, Platform} from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import * as posedetection from '@tensorflow-models/pose-detection';
import {cameraWithTensors } from '@tensorflow/tfjs-react-native';
import Svg, { Circle } from 'react-native-svg';
import { ExpoWebGLRenderingContext } from 'expo-gl';
import { CameraType } from 'expo-camera/build/Camera.types';
import * as ScreenOrientation from 'expo-screen-orientation';

const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === 'android';
const IS_IOS = Platform.OS === 'ios';
const CAM_PREVIEW_WIDTH = Dimensions.get('window').width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);
const MIN_KEYPOINT_SCORE = 0.3;
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);
const AUTO_RENDER = false;
const LOAD_MODEL_FROM_BUNDLE = false;

export default function App() {
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  
  const cameraRef = useRef()
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState();
  const [poses, setPoses] = useState();
  const [fps, setFps] = useState(0);
  const [orientation, setOrientation] = useState();
  const [cameraType, setCameraType] = useState(Camera.Constants.Type.front);
  const rafId = useRef(null)

  useEffect(() => {
    async function prepare() {
      rafId.current = null;
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });
      
      await Camera.requestCameraPermissionsAsync();
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
      const movenetModelConfig = {
        modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
      };
      // if (LOAD_MODEL_FROM_BUNDLE) {
      //   const modelJson = require('./offline_model/model.json');
      //   const modelWeights1 = require('./offline_model/group1-shard1of2.bin');
      //   const modelWeights2 = require('./offline_model/group1-shard2of2.bin');
      //   movenetModelConfig.modelUrl = bundleResourceIO(modelJson, [
      //     modelWeights1,
      //     modelWeights2,
      //   ]);
      // }
      const model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        movenetModelConfig
      );
      setModel(model);
      console.log('Model loaded')

      // Ready!
      setTfReady(true);
    }
    prepare();
  }, [])

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const handleCameraStream = async (images, updatePreview, gl) => {
    const loop = async () => {

        // Get the tensor and run pose detection.
        const imageTensor = images.next().value;

        const startTs = Date.now();
        const poses = await model.estimatePoses(
            imageTensor,
            undefined,
            Date.now()
        );
        const latency = Date.now() - startTs;
        setFps(Math.floor(1000 / latency));
        setPoses(poses);
        tf.dispose([imageTensor]);

        if (rafId.current === 0) {
            return;
        }

        // Render camera preview manually when autorender=false.
        if (!AUTO_RENDER) {
            updatePreview();
            gl.endFrameEXP();
        }

        rafId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  const renderPose = () => {
      if (poses != null && poses.length > 0) {
          const keypoints = poses[0].keypoints
              .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
              .map((k) => {
                  // Flip horizontally on android or when using back camera on iOS.
                  const flipX = IS_ANDROID || cameraType === Camera.Constants.Type.back;
                  const x = flipX ? getOutputTensorWidth() - k.x : k.x;
                  const y = k.y;
                  const cx =
                      (x / getOutputTensorWidth()) *
                      (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
                  const cy =
                      (y / getOutputTensorHeight()) *
                      (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
                  return (
                      <Circle
                          key={`skeletonkp_${k.name}`}
                          cx={cx}
                          cy={cy}
                          r='4'
                          strokeWidth='2'
                          fill='#00AA00'
                          stroke='white'
                      />
                  );
              });

          return <Svg style={styles.svg}>{keypoints}</Svg>;
      } else {
          return <View></View>;
      }
  };

  const renderFps = () => {
      return (
          <View style={styles.fpsContainer}>
              <Text>FPS: {fps}</Text>
          </View>
      );
  };

  const renderCameraTypeSwitcher = () => {
      return (
          <View
              style={styles.cameraTypeSwitcher}
              onTouchEnd={handleSwitchCameraType}
          >
              <Text>
                  Switch to{' '}
                  {cameraType === Camera.Constants.Type.front ? 'back' : 'front'} camera
              </Text>
          </View>
      );
  };

  const handleSwitchCameraType = () => {
      if (cameraType === Camera.Constants.Type.front) {
          setCameraType(Camera.Constants.Type.back);
      } else {
          setCameraType(Camera.Constants.Type.front);
      }
  };

  const isPortrait = () => {
      return (
          orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
          orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
      );
  };

  const getOutputTensorWidth = () => {
      // On iOS landscape mode, switch width and height of the output tensor to
      // get better result. Without this, the image stored in the output tensor
      // would be stretched too much.
      //
      // Same for getOutputTensorHeight below.
      return isPortrait() || IS_ANDROID
        ? OUTPUT_TENSOR_WIDTH
        : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
      return isPortrait() || IS_ANDROID
        ? OUTPUT_TENSOR_HEIGHT
        : OUTPUT_TENSOR_WIDTH;
    };

  const getTextureRotationAngleInDegrees = () => {
      // On Android, the camera texture will rotate behind the scene as the phone
      // changes orientation, so we don't need to rotate it in TensorCamera.
      // if (IS_ANDROID) {
      //   console.log('android')
      //   return 0;
      // }

      // // For iOS, the camera texture won't rotate automatically. Calculate the
      // // rotation angles here which will be passed to TensorCamera to rotate it
      // // internally.
      // switch (orientation) {
      //     // Not supported on iOS as of 11/2021, but add it here just in case.
      //     case ScreenOrientation.Orientation.PORTRAIT_DOWN:
      //         return 180;
      //     case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
      //         return cameraType === Camera.Constants.Type.front ? 270 : 90;
      //     case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
      //         return cameraType === Camera.Constants.Type.front ? 90 : 270;
      //     default:
      //         return 0;
      // }
      return 0;
  };

  // if (hasPermission === null) {
  //   return <View />;
  // }
  // if (hasPermission === false) {
  //   return <Text>No access to camera</Text>;
  // }
  return (
      <View style={isPortrait() ? styles.containerPortrait : styles.containerLandscape}>
        <TensorCamera
            ref={cameraRef}
            style={styles.camera}
            type={cameraType}
            onReady={handleCameraStream}
            // tensor related props
            resizeWidth={4}
            resizeHeight={getOutputTensorHeight}
            resizeDepth={3}
            rotation={0}
        />
        {renderPose()}
        {renderFps()}
        {renderCameraTypeSwitcher()}
    </View>
);
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    width: '100%',
    height: '100%',
    zIndex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    margin: 20,
  },
  button: {
    flex: 0.1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 18,
    color: 'white',
  },
  svg: {
    width: '100%',
    height: '100%',
    position: 'absolute',
    zIndex: 30,
  },
  fpsContainer: {
    position: 'absolute',
    top: 10,
    left: 10,
    width: 80,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 180,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, .7)',
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  containerPortrait: {
    position: 'relative',
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: 'relative',
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
