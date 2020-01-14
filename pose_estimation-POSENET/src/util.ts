/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import { connectedPartIndeces } from './keypoints';
import { OutputStride } from './mobilenet';
import { Keypoint, Pose, TensorBuffer3D, Vector2D } from './types';

function eitherPointDoesntMeetConfidence(
  a: number, b: number, minConfidence: number): boolean {
  return (a < minConfidence || b < minConfidence);
}

export function getAdjacentKeyPoints(
  keypoints: Keypoint[], minConfidence: number): Keypoint[][] {
  return connectedPartIndeces.reduce(
    (result: Keypoint[][], [leftJoint, rightJoint]): Keypoint[][] => {
      if (eitherPointDoesntMeetConfidence(
        keypoints[leftJoint].score, keypoints[rightJoint].score,
        minConfidence)) {
        return result;
      }

      result.push([keypoints[leftJoint], keypoints[rightJoint]]);

      return result;
    }, []);
}

const { NEGATIVE_INFINITY, POSITIVE_INFINITY } = Number;
export function getBoundingBox(keypoints: Keypoint[]): { maxX: number, maxY: number, minX: number, minY: number } {
  return keypoints.reduce(({ maxX, maxY, minX, minY }, { position: { x, y } }) => {
    return {
      maxX: Math.max(maxX, x),
      maxY: Math.max(maxY, y),
      minX: Math.min(minX, x),
      minY: Math.min(minY, y)
    };
  }, {
      maxX: NEGATIVE_INFINITY,
      maxY: NEGATIVE_INFINITY,
      minX: POSITIVE_INFINITY,
      minY: POSITIVE_INFINITY
    });
}

export function getBoundingBoxPoints(keypoints: Keypoint[]): Vector2D[] {
  const { minX, minY, maxX, maxY } = getBoundingBox(keypoints);
  return [
    { x: minX, y: minY }, { x: maxX, y: minY }, { x: maxX, y: maxY },
    { x: minX, y: maxY }
  ];
}

export async function toTensorBuffer<rank extends tf.Rank>(
  tensor: tf.Tensor<rank>,
  type: 'float32' | 'int32' = 'float32'): Promise<tf.TensorBuffer<rank>> {
  const tensorData = await tensor.data();

  return new tf.TensorBuffer<rank>(tensor.shape, type, tensorData);
}

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
  Promise<TensorBuffer3D[]> {
  return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
}

export function scalePose(pose: Pose, scaleX: number, scaleY: number): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(
      ({ score, part, position }) => ({
        score,
        part,
        position: { x: position.x * scaleX, y: position.y * scaleY }
      }))
  };
}

export function scalePoses(poses: Pose[], scaleY: number, scaleX: number): Pose[] {
  if (scaleX === 1 && scaleY === 1) {
    return poses;
  }
  return poses.map(pose => scalePose(pose, scaleX, scaleY));
}

export function getValidResolution(
  imageScaleFactor: number, inputDimension: number,
  outputStride: OutputStride): number {
  const evenResolution = inputDimension * imageScaleFactor - 1;

  return evenResolution - (evenResolution % outputStride) + 1;
}
